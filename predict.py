from joblib import load
import json
import numpy as np
import pandas as pd
from os import path
import random
from feature_engineering import add_features_to_match

def load_models():
    # Load the models and metadata
    lr_classifier = load('exportedModels/lr_classifier.model')
    nb_classifier = load('exportedModels/nb_classifier.model')
    rf_classifier = load('exportedModels/rf_classifier.model')
    
    with open('exportedModels/metaData.json', 'r') as f:
        metadata = json.load(f)
        
    with open('exportedModels/team_stats.json', 'r') as f:
        team_stats = json.load(f)
    
    return lr_classifier, nb_classifier, rf_classifier, metadata, team_stats

def load_match_data():
    # Data folders for different leagues
    data_folders = [
        'english-premier-league_zip',
        'spanish-la-liga_zip',
        'french-ligue-1_zip',
        'german-bundesliga_zip',
        'italian-serie-a_zip'
    ]
    
    # Load a random season between 2009-2018
    season = random.randint(9, 18)
    data_folder = random.choice(data_folders)
    
    data_file = f'data/{data_folder}/data/season-{season:02d}{season+1:02d}_csv.csv'
    
    if not path.exists(data_file):
        print(f"Error: Could not find data file {data_file}")
        return None
        
    data = pd.read_csv(data_file)
    
    # Select a random match from the dataset
    match = data.sample(n=1).iloc[0]
    
    return match

def predict_match_result(match_data, models):
    lr_classifier, nb_classifier, rf_classifier, metadata, team_stats = models
    
    home_team = match_data['HomeTeam']
    away_team = match_data['AwayTeam']
    
    # Get team encodings from metadata
    home_encoded = metadata['home_teams'].get(home_team)
    away_encoded = metadata['away_teams'].get(away_team)
    
    if home_encoded is None or away_encoded is None:
        print(f"Error: Team(s) not found in training data")
        print(f"Available home teams: {list(metadata['home_teams'].keys())}")
        print(f"Available away teams: {list(metadata['away_teams'].keys())}")
        return None
    
    # Get additional features
    additional_features = add_features_to_match(match_data, team_stats)
    
    # Create feature vector with half-time stats and additional features
    X = np.array([[
        home_encoded,
        away_encoded,
        match_data['HTHG'],  # Half-time home goals
        match_data['HTAG'],  # Half-time away goals
        match_data['HS'],    # Home shots
        match_data['AS'],    # Away shots
        match_data['HST'],   # Home shots on target
        match_data['AST'],   # Away shots on target
        match_data['HR'],    # Home red cards
        match_data['AR'],    # Away red cards
        additional_features['home_win_ratio'],
        additional_features['away_win_ratio'],
        additional_features['home_team_home_win_ratio'],
        additional_features['away_team_away_win_ratio'],
        additional_features['home_goals_per_game'],
        additional_features['away_goals_per_game'],
        additional_features['home_comeback_ratio'],
        additional_features['away_comeback_ratio'],
        additional_features['home_second_half_goal_ratio'],
        additional_features['away_second_half_goal_ratio']
    ]])
    
    # Get predictions from all models
    lr_pred = lr_classifier.predict(X)[0]
    nb_pred = nb_classifier.predict(X)[0]
    rf_pred = rf_classifier.predict(X)[0]
    
    # Get prediction probabilities
    lr_proba = lr_classifier.predict_proba(X)[0]
    nb_proba = nb_classifier.predict_proba(X)[0]
    rf_proba = rf_classifier.predict_proba(X)[0]
    
    print("\nMatch Details:")
    print("-" * 50)
    print(f"League: {match_data['Div']}")
    print(f"{home_team} vs {away_team}")
    print(f"Half-time score: {int(match_data['HTHG'])} - {int(match_data['HTAG'])}")
    print(f"Shots (on target): {home_team}: {int(match_data['HS'])} ({int(match_data['HST'])}), {away_team}: {int(match_data['AS'])} ({int(match_data['AST'])})")
    print(f"Red cards: {home_team}: {int(match_data['HR'])}, {away_team}: {int(match_data['AR'])}")
    
    print("\nTeam Statistics:")
    print("-" * 50)
    print(f"{home_team}:")
    print(f"- Overall win ratio: {additional_features['home_win_ratio']:.3f}")
    print(f"- Home win ratio: {additional_features['home_team_home_win_ratio']:.3f}")
    print(f"- Goals per game: {additional_features['home_goals_per_game']:.2f}")
    print(f"- Comeback ratio: {additional_features['home_comeback_ratio']:.3f}")
    print(f"- Second half goal ratio: {additional_features['home_second_half_goal_ratio']:.3f}")
    
    print(f"\n{away_team}:")
    print(f"- Overall win ratio: {additional_features['away_win_ratio']:.3f}")
    print(f"- Away win ratio: {additional_features['away_team_away_win_ratio']:.3f}")
    print(f"- Goals per game: {additional_features['away_goals_per_game']:.2f}")
    print(f"- Comeback ratio: {additional_features['away_comeback_ratio']:.3f}")
    print(f"- Second half goal ratio: {additional_features['away_second_half_goal_ratio']:.3f}")
    
    print(f"\nActual full-time result: {match_data['FTR']} (Score: {int(match_data['FTHG'])} - {int(match_data['FTAG'])})")
    
    print("\nPredictions:")
    print("-" * 50)
    print(f"Logistic Regression predicts: {lr_pred} (H: {lr_proba[0]:.2f}, D: {lr_proba[1]:.2f}, A: {lr_proba[2]:.2f})")
    print(f"Naive Bayes predicts: {nb_pred} (H: {nb_proba[0]:.2f}, D: {nb_proba[1]:.2f}, A: {nb_proba[2]:.2f})")
    print(f"Random Forest predicts: {rf_pred} (H: {rf_proba[0]:.2f}, D: {rf_proba[1]:.2f}, A: {rf_proba[2]:.2f})")

def main():
    print("Loading models...")
    models = load_models()
    
    print("Loading random match data...")
    match_data = load_match_data()
    
    if match_data is not None:
        predict_match_result(match_data, models)

if __name__ == "__main__":
    main() 