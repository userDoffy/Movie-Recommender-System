from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movies dataset
movies = pd.read_csv('movies.csv')
movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' ') if isinstance(x, str) else '')
movies['title_normalized'] = movies['title'].str.lower().str.strip()

# Preprocess genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create reverse mapping for titles
movie_indices = pd.Series(movies.index, index=movies['title_normalized']).drop_duplicates()

# Search function
def search_movies(query):
    query = query.lower().strip()
    results = movies[movies['title_normalized'].str.contains(query)]
    return results[['title', 'genres']].to_dict(orient='records')

# Recommendation function
def recommend_movies(selected_movie_ids):
    if not selected_movie_ids:
        return []

    combined_scores = sum(cosine_sim[movie_id] for movie_id in selected_movie_ids) / len(selected_movie_ids)

    recommended_indices = combined_scores.argsort()[-21:][::-1]
    recommended_indices = [i for i in recommended_indices if i not in selected_movie_ids][:20]

    return movies.iloc[recommended_indices][['title', 'genres']].to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    search_results = search_movies(query)
    return jsonify(search_results)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movies = request.json.get('selected_movies', [])
    selected_movie_ids = [movie_indices.get(movie.lower().strip()) for movie in selected_movies]
    if None in selected_movie_ids:
        return jsonify({"error": "One or more selected movies not found."}), 400
    recommendations = recommend_movies(selected_movie_ids)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
