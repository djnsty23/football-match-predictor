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

def prepare_prediction_features(home_team, away_team, home_goals=None, away_goals=None, team_encoder=None, feature_names=None):
    """Prepare features for prediction"""
    try:
        # Encode teams
        home_encoded = team_encoder.transform([home_team])[0]
        away_encoded = team_encoder.transform([away_team])[0]
        
        # Create feature dictionary with default values
        feature_dict = {
            'home_encoded': home_encoded,
            'away_encoded': away_encoded,
            'HTHG': float(home_goals) if home_goals is not None else 0.0,
            'HTAG': float(away_goals) if away_goals is not None else 0.0,
            'HS': 0.0,  # Will be provided during match
            'AS': 0.0,
            'HST': 0.0,
            'AST': 0.0,
            'HR': 0.0,
            'AR': 0.0,
            'HomeElo': 1500.0,  # Default Elo rating
            'AwayElo': 1500.0,
            'home_form_points': 0.0,
            'away_form_points': 0.0,
            'home_goals_scored_form': 0.0,
            'away_goals_scored_form': 0.0,
            'home_goals_conceded_form': 0.0,
            'away_goals_conceded_form': 0.0,
            'home_clean_sheets_form': 0.0,
            'away_clean_sheets_form': 0.0,
            'home_days_rest': 7.0,
            'away_days_rest': 7.0,
            'home_league_position': 10.0,
            'away_league_position': 10.0,
            'home_points_per_game': 1.5,
            'away_points_per_game': 1.5,
            'home_goal_diff_per_game': 0.0,
            'away_goal_diff_per_game': 0.0
        }
        
        # Create DataFrame with features in the correct order
        X = pd.DataFrame([feature_dict])
        
        # Ensure all required features are present and in the correct order
        if feature_names:
            # Add missing columns with default value 0
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            # Select only the features used in training, in the same order
            X = X[feature_names]
        
        return X
        
    except Exception as e:
        raise Exception(f"Error preparing prediction features: {str(e)}")

def predict_match_result(home_team, away_team, home_goals=None, away_goals=None):
    try:
        # Load models and metadata
        lr_classifier = load('exportedModels/lr_classifier_enhanced.model')
        nb_classifier = load('exportedModels/nb_classifier_enhanced.model')
        rf_classifier = load('exportedModels/rf_classifier_enhanced.model')
        scaler = load('exportedModels/feature_scaler.model')
        team_encoder = load('exportedModels/team_encoder.model')

        with open('exportedModels/metadata_enhanced.json', 'r') as f:
            metadata = json.load(f)

        # Prepare features
        features = prepare_prediction_features(home_team, away_team, home_goals, away_goals, team_encoder, metadata['features'])
        X_scaled = scaler.transform(features)

        # Get predictions from each model
        # Models return string predictions ('H', 'D', 'A')
        lr_pred = lr_classifier.predict(X_scaled)[0]  # This is a string ('H', 'D', or 'A')
        nb_pred = nb_classifier.predict(X_scaled)[0]  # This is a string ('H', 'D', or 'A')
        rf_pred = rf_classifier.predict(X_scaled)[0]  # This is a string ('H', 'D', or 'A')

        # Get prediction probabilities
        lr_probs = lr_classifier.predict_proba(X_scaled)[0]
        nb_probs = nb_classifier.predict_proba(X_scaled)[0]
        rf_probs = rf_classifier.predict_proba(X_scaled)[0]

        # The predictions are already strings, no need to convert
        lr_pred_str = lr_pred
        nb_pred_str = nb_pred
        rf_pred_str = rf_pred

        # Calculate weighted probabilities for each outcome
        weighted_probs = {
            'H': 0.3 * lr_probs[0] + 0.3 * nb_probs[0] + 0.4 * rf_probs[0],
            'D': 0.3 * lr_probs[1] + 0.3 * nb_probs[1] + 0.4 * rf_probs[1],
            'A': 0.3 * lr_probs[2] + 0.3 * nb_probs[2] + 0.4 * rf_probs[2]
        }

        # Get ensemble prediction (outcome with highest weighted probability)
        ensemble_pred = max(weighted_probs.items(), key=lambda x: x[1])[0]

        return {
            'predictions': {
                'logistic_regression': lr_pred_str,
                'naive_bayes': nb_pred_str,
                'random_forest': rf_pred_str,
                'ensemble': ensemble_pred
            },
            'probabilities': weighted_probs
        }

    except Exception as e:
        raise Exception(f"Error in predict_match_result: {str(e)}")

def main():
    print("Loading models...")
    models = load_models()
    
    print("Loading random match data...")
    match_data = load_match_data()
    
    if match_data is not None:
        predict_match_result(match_data, models)

if __name__ == "__main__":
    main() 