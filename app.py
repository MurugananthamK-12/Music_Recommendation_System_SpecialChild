from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv(r"C:\Users\murug\Desktop\Music_Recommendation_System_SpecialChild-main\(edited)dataset.csv")
sd = df.iloc[:5000,:]

sd = sd.drop_duplicates(subset=['artists','track_name'],keep=False)

# Normalize features
scaler = MinMaxScaler()
sd[['loudness', 'instrumentalness', 'valence', 'energy', 'danceability', 'tempo', 'acousticness']] = scaler.fit_transform(sd[['loudness', 'instrumentalness', 'valence', 'energy', 'danceability', 'tempo', 'acousticness']])

sd1 = sd.drop(columns=['track_id', 'popularity'], axis=1)

condition_features = {
    'asd': ['loudness', 'instrumentalness', 'valence', 'energy', 'danceability', 'tempo'],
    'adhd': ['energy', 'tempo'],
    'down_syndrome': ['valence', 'danceability', 'tempo'],
    'sensory_processing_disorder': ['acousticness', 'loudness', 'tempo']
}

def recommended_songs(track_name, condition, num_rec=5):
    idx = sd1.index[sd1['track_name'] == track_name].tolist()
    if not idx:
        return pd.DataFrame()  # Return an empty DataFrame if track_name not found
    idx = idx[0]
    features = condition_features.get(condition, [])
    cosine_sim = cosine_similarity(sd1[features])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    song_indices = [i[0] for i in sim_scores[1:num_rec+1]]
    return df.iloc[song_indices][['artists', 'track_name']]

@app.route('/recommend', methods=['GET'])
def recommend():
    track_name = request.args.get('track_name')
    condition = request.args.get('condition')
    num_rec = int(request.args.get('num_rec', 5))
    
    if not track_name or not condition:
        return jsonify({'error': 'Missing track_name or condition'}), 400
    
    recommendations = recommended_songs(track_name, condition, num_rec)
    if recommendations.empty:
        return jsonify({'error': 'Track not found'}), 404
    
    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    # Filter unique track names that contain the query
    matches = sd[sd['track_name'].str.lower().str.contains(query, na=False)]['track_name'].unique()
    return jsonify(list(matches[:10]))  # Limit to top 10 suggestions

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
