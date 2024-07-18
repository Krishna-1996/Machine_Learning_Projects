from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

app = Flask(__name__)

# Load model and encoders
model = keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')
venue_encoder = joblib.load('venue_encoder.pkl')
bat_team_encoder = joblib.load('bat_team_encoder.pkl')
bowl_team_encoder = joblib.load('bowl_team_encoder.pkl')
batsman_encoder = joblib.load('batsman_encoder.pkl')
bowler_encoder = joblib.load('bowler_encoder.pkl')

# Load dataset for dropdown options
ipl = pd.read_csv('E:/Machine Learning/Machine Learning/Machine_Learning_Projects/Machine_Learning_Projects/1. IPL Score Prediction/ipl_data.csv')

@app.route('/')
def index():
    venues = ipl['venue'].unique().tolist()
    bat_teams = ipl['bat_team'].unique().tolist()
    bowl_teams = ipl['bowl_team'].unique().tolist()
    batsmen = ipl['batsman'].unique().tolist()
    bowlers = ipl['bowler'].unique().tolist()
    return render_template('index.html', venues=venues, bat_teams=bat_teams, bowl_teams=bowl_teams, batsmen=batsmen, bowlers=bowlers)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    venue = data['venue']
    bat_team = data['bat_team']
    bowl_team = data['bowl_team']
    batsman = data['batsman']
    bowler = data['bowler']
    
    input_data = np.array([
        venue_encoder.transform([venue])[0],
        bat_team_encoder.transform([bat_team])[0],
        bowl_team_encoder.transform([bowl_team])[0],
        batsman_encoder.transform([batsman])[0],
        bowler_encoder.transform([bowler])[0]
    ]).reshape(1, -1)
    
    input_data = scaler.transform(input_data)
    predicted_score = model.predict(input_data)[0][0]
    
    return jsonify({'predicted_score': int(predicted_score)})

if __name__ == '__main__':
    app.run(debug=True)
model.save('model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(venue_encoder, 'venue_encoder.pkl')
joblib.dump(bat_team_encoder, 'bat_team_encoder.pkl')
joblib.dump(bowl_team_encoder, 'bowl_team_encoder.pkl')
joblib.dump(batsman_encoder, 'batsman_encoder.pkl')
joblib.dump(bowler_encoder, 'bowler_encoder.pkl')
