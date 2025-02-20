from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from time import time
from sklearn.metrics import f1_score, accuracy_score, classification_report
from os import path, makedirs
from joblib import dump
import json
from feature_engineering import calculate_team_stats, add_features_to_match, load_all_data
import sys
from tqdm import tqdm

def calculate_elo_ratings(matches, k_factor=32, initial_rating=1500):
    """Calculate Elo ratings for teams based on historical performance"""
    elo_ratings = {}
    match_ratings = []

    for _, match in matches.iterrows():
        # Get or initialize ratings
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        if home_team not in elo_ratings:
            elo_ratings[home_team] = initial_rating
        if away_team not in elo_ratings:
            elo_ratings[away_team] = initial_rating
        
        # Calculate expected scores
        rating_diff = (elo_ratings[home_team] - elo_ratings[away_team]) / 400
        expected_home = 1 / (1 + 10 ** (-rating_diff))
        expected_away = 1 - expected_home
        
        # Calculate actual scores
        if match['FTR'] == 'H':
            actual_home = 1
            actual_away = 0
        elif match['FTR'] == 'A':
            actual_home = 0
            actual_away = 1
        else:  # Draw
            actual_home = 0.5
            actual_away = 0.5
        
        # Store pre-match ratings
        match_ratings.append({
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'HomeElo': elo_ratings[home_team],
            'AwayElo': elo_ratings[away_team]
        })
        
        # Update ratings
        elo_ratings[home_team] += k_factor * (actual_home - expected_home)
        elo_ratings[away_team] += k_factor * (actual_away - expected_away)
    
    return pd.DataFrame(match_ratings)

def calculate_form_features(matches):
    """Calculate rolling form features for each team"""
    form_features = []
    
    # Sort matches by date
    matches = matches.sort_values('Date')
    
    # Initialize team form dictionaries
    team_form = {}
    
    for _, match in matches.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Initialize if team not seen before
        if home_team not in team_form:
            team_form[home_team] = {
                'last_5_results': [],
                'last_5_goals_for': [],
                'last_5_goals_against': [],
                'last_5_clean_sheets': 0,
                'days_since_last_match': None,
                'last_match_date': None
            }
        if away_team not in team_form:
            team_form[away_team] = {
                'last_5_results': [],
                'last_5_goals_for': [],
                'last_5_goals_against': [],
                'last_5_clean_sheets': 0,
                'days_since_last_match': None,
                'last_match_date': None
            }
        
        # Calculate days since last match
        if team_form[home_team]['last_match_date']:
            team_form[home_team]['days_since_last_match'] = (match['Date'] - team_form[home_team]['last_match_date']).days
        if team_form[away_team]['last_match_date']:
            team_form[away_team]['days_since_last_match'] = (match['Date'] - team_form[away_team]['last_match_date']).days
        
        # Store current form
        home_form = {
            'last_5_results': team_form[home_team]['last_5_results'][-5:],
            'last_5_goals_for': team_form[home_team]['last_5_goals_for'][-5:],
            'last_5_goals_against': team_form[home_team]['last_5_goals_against'][-5:],
            'last_5_clean_sheets': team_form[home_team]['last_5_clean_sheets'],
            'days_since_last_match': team_form[home_team]['days_since_last_match']
        }
        
        away_form = {
            'last_5_results': team_form[away_team]['last_5_results'][-5:],
            'last_5_goals_for': team_form[away_team]['last_5_goals_for'][-5:],
            'last_5_goals_against': team_form[away_team]['last_5_goals_against'][-5:],
            'last_5_clean_sheets': team_form[away_team]['last_5_clean_sheets'],
            'days_since_last_match': team_form[away_team]['days_since_last_match']
        }
        
        form_features.append({
            'home_form_points': sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in home_form['last_5_results']]) / (3 * len(home_form['last_5_results'])) if home_form['last_5_results'] else 0,
            'away_form_points': sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in away_form['last_5_results']]) / (3 * len(away_form['last_5_results'])) if away_form['last_5_results'] else 0,
            'home_goals_scored_form': sum(home_form['last_5_goals_for']) / len(home_form['last_5_goals_for']) if home_form['last_5_goals_for'] else 0,
            'away_goals_scored_form': sum(away_form['last_5_goals_for']) / len(away_form['last_5_goals_for']) if away_form['last_5_goals_for'] else 0,
            'home_goals_conceded_form': sum(home_form['last_5_goals_against']) / len(home_form['last_5_goals_against']) if home_form['last_5_goals_against'] else 0,
            'away_goals_conceded_form': sum(away_form['last_5_goals_against']) / len(away_form['last_5_goals_against']) if away_form['last_5_goals_against'] else 0,
            'home_clean_sheets_form': home_form['last_5_clean_sheets'] / 5 if home_form['last_5_results'] else 0,
            'away_clean_sheets_form': away_form['last_5_clean_sheets'] / 5 if away_form['last_5_results'] else 0,
            'home_days_rest': home_form['days_since_last_match'] if home_form['days_since_last_match'] is not None else 7,
            'away_days_rest': away_form['days_since_last_match'] if away_form['days_since_last_match'] is not None else 7
        })
        
        # Update form after match
        # Home team
        result = 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L'
        team_form[home_team]['last_5_results'].append(result)
        team_form[home_team]['last_5_goals_for'].append(match['FTHG'])
        team_form[home_team]['last_5_goals_against'].append(match['FTAG'])
        if match['FTAG'] == 0:
            team_form[home_team]['last_5_clean_sheets'] = (team_form[home_team]['last_5_clean_sheets'] * 4 + 1) / 5
        else:
            team_form[home_team]['last_5_clean_sheets'] = (team_form[home_team]['last_5_clean_sheets'] * 4) / 5
        team_form[home_team]['last_match_date'] = match['Date']
        
        # Away team
        result = 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L'
        team_form[away_team]['last_5_results'].append(result)
        team_form[away_team]['last_5_goals_for'].append(match['FTAG'])
        team_form[away_team]['last_5_goals_against'].append(match['FTHG'])
        if match['FTHG'] == 0:
            team_form[away_team]['last_5_clean_sheets'] = (team_form[away_team]['last_5_clean_sheets'] * 4 + 1) / 5
        else:
            team_form[away_team]['last_5_clean_sheets'] = (team_form[away_team]['last_5_clean_sheets'] * 4) / 5
        team_form[away_team]['last_match_date'] = match['Date']
    
    return pd.DataFrame(form_features)

def calculate_league_position_features(matches):
    """Calculate league position and points for teams throughout the season"""
    position_features = []
    
    # Group matches by season and process each season separately
    for season in matches['Season'].unique():
        season_matches = matches[matches['Season'] == season].sort_values('Date')
        
        # Initialize season standings
        standings = {}
        
        for _, match in season_matches.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Initialize teams if not in standings
            if home_team not in standings:
                standings[home_team] = {'points': 0, 'played': 0, 'goal_diff': 0}
            if away_team not in standings:
                standings[away_team] = {'points': 0, 'played': 0, 'goal_diff': 0}
            
            # Store pre-match standings
            teams = list(standings.keys())
            sorted_teams = sorted(teams, key=lambda x: (-standings[x]['points'], 
                                                      -standings[x]['goal_diff'], 
                                                      -standings[x]['played']))
            positions = {team: pos+1 for pos, team in enumerate(sorted_teams)}
            
            position_features.append({
                'home_league_position': positions.get(home_team, len(standings)),
                'away_league_position': positions.get(away_team, len(standings)),
                'home_points_per_game': standings[home_team]['points'] / standings[home_team]['played'] if standings[home_team]['played'] > 0 else 0,
                'away_points_per_game': standings[away_team]['points'] / standings[away_team]['played'] if standings[away_team]['played'] > 0 else 0,
                'home_goal_diff_per_game': standings[home_team]['goal_diff'] / standings[home_team]['played'] if standings[home_team]['played'] > 0 else 0,
                'away_goal_diff_per_game': standings[away_team]['goal_diff'] / standings[away_team]['played'] if standings[away_team]['played'] > 0 else 0
            })
            
            # Update standings after match
            standings[home_team]['played'] += 1
            standings[away_team]['played'] += 1
            
            if match['FTR'] == 'H':
                standings[home_team]['points'] += 3
            elif match['FTR'] == 'A':
                standings[away_team]['points'] += 3
            else:
                standings[home_team]['points'] += 1
                standings[away_team]['points'] += 1
            
            standings[home_team]['goal_diff'] += match['FTHG'] - match['FTAG']
            standings[away_team]['goal_diff'] += match['FTAG'] - match['FTHG']
    
    return pd.DataFrame(position_features)

def prepare_enhanced_features(matches):
    """Prepare enhanced feature set for model training"""
    # Clean the data first
    matches = matches.copy()
    
    # Convert date strings to datetime
    matches['Date'] = pd.to_datetime(matches['Date'])
    
    # Fill missing values in essential columns
    essential_numeric_columns = ['HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR']
    for col in essential_numeric_columns:
        if col in matches.columns:
            matches[col] = matches[col].fillna(0)
    
    print("Calculating Elo ratings...")
    elo_features = calculate_elo_ratings(matches)
    
    print("Calculating form features...")
    form_features = calculate_form_features(matches)
    
    print("Calculating league position features...")
    position_features = calculate_league_position_features(matches)
    
    # Combine all features
    enhanced_features = pd.concat([
        matches[['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR', 'FTR']],
        elo_features[['HomeElo', 'AwayElo']],
        form_features,
        position_features
    ], axis=1)
    
    # Fill any remaining NaN values with appropriate defaults
    numeric_columns = enhanced_features.select_dtypes(include=[np.number]).columns
    enhanced_features[numeric_columns] = enhanced_features[numeric_columns].fillna(0)
    
    # Drop any rows that still have NaN values
    enhanced_features = enhanced_features.dropna()
    
    return enhanced_features

def train_classifier(clf, X_train, y_train):
    try:
        start = time()
        clf.fit(X_train, y_train)
        end = time()
        print("Model trained in {:.2f} seconds".format(end-start))
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def predict_labels(clf, features, target):
    try:
        start = time()
        y_pred = clf.predict(features)
        end = time()
        print("Made Predictions in {:.2f} seconds".format(end-start))

        acc = accuracy_score(target, y_pred)
        f1 = f1_score(target, y_pred, average='weighted')
        return f1, acc
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def model(clf, X_train, y_train, X_test, y_test, print_report=False):
    try:
        train_classifier(clf, X_train, y_train)

        f1, acc = predict_labels(clf, X_train, y_train)
        print("Training Info:")
        print("-" * 20)
        print("F1 Score: {:.3f}".format(f1))
        print("Accuracy: {:.3f}".format(acc))

        f1, acc = predict_labels(clf, X_test, y_test)
        print("Test Metrics:")
        print("-" * 20)
        print("F1 Score: {:.3f}".format(f1))
        print("Accuracy: {:.3f}".format(acc))
        
        if print_report:
            print("\nDetailed Classification Report:")
            print("-" * 50)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise

def tune_random_forest(X_train, y_train, X_test, y_test):
    """Perform grid search to find the best Random Forest parameters"""
    try:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1,
            scoring='f1_weighted',
            verbose=1
        )
        
        print("Performing grid search for Random Forest...")
        grid_search.fit(X_train, y_train)
        
        print("\nBest parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
        
        best_rf = grid_search.best_estimator_
        print("\nEvaluating best model:")
        model(best_rf, X_train, y_train, X_test, y_test, print_report=True)
        
        return best_rf
    except Exception as e:
        print(f"Error during Random Forest tuning: {str(e)}")
        raise

def main():
    try:
        print("Loading all match data...")
        data = load_all_data()
        
        # Drop rows with missing essential values
        essential_columns = ['HomeTeam', 'AwayTeam', 'FTR', 'Date']
        data = data.dropna(subset=essential_columns)
        
        print("Preparing enhanced features...")
        enhanced_features = prepare_enhanced_features(data)
        
        print(f"Total matches after cleaning: {len(enhanced_features)}")
        print("\nFeature summary:")
        print(enhanced_features.describe())
        
        print("\nMissing values:")
        print(enhanced_features.isnull().sum())
        
        print("Encoding categorical variables...")
        # Encode team names
        team_encoder = LabelEncoder()
        enhanced_features['home_encoded'] = team_encoder.fit_transform(enhanced_features['HomeTeam'])
        enhanced_features['away_encoded'] = team_encoder.fit_transform(enhanced_features['AwayTeam'])
        
        # Create feature matrix
        feature_columns = [
            'home_encoded', 'away_encoded',
            'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR',
            'HomeElo', 'AwayElo',
            'home_form_points', 'away_form_points',
            'home_goals_scored_form', 'away_goals_scored_form',
            'home_goals_conceded_form', 'away_goals_conceded_form',
            'home_clean_sheets_form', 'away_clean_sheets_form',
            'home_days_rest', 'away_days_rest',
            'home_league_position', 'away_league_position',
            'home_points_per_game', 'away_points_per_game',
            'home_goal_diff_per_game', 'away_goal_diff_per_game'
        ]
        
        X = enhanced_features[feature_columns]
        y = enhanced_features['FTR']
        
        # Verify no missing values
        assert not X.isnull().any().any(), "Found missing values in features"
        assert not y.isnull().any(), "Found missing values in target"
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Use time-based split instead of random split
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print("\nTraining enhanced models...")
        
        print("\nLogistic Regression one vs All Classifier")
        print("-" * 50)
        lr_classifier = LogisticRegression(multi_class='ovr', max_iter=1000, class_weight='balanced')
        model(lr_classifier, X_train, y_train, X_test, y_test)

        print("\nGaussian Naive Bayes Classifier")
        print("-" * 50)
        nb_classifier = GaussianNB()
        model(nb_classifier, X_train, y_train, X_test, y_test)

        print("\nTuning Random Forest Classifier")
        print("-" * 50)
        rf_classifier = tune_random_forest(X_train, y_train, X_test, y_test)
        
        print("\nExporting enhanced models...")
        export_dir = 'exportedModels'
        makedirs(export_dir, exist_ok=True)

        dump(lr_classifier, f'{export_dir}/lr_classifier_enhanced.model')
        dump(nb_classifier, f'{export_dir}/nb_classifier_enhanced.model')
        dump(rf_classifier, f'{export_dir}/rf_classifier_enhanced.model')
        dump(scaler, f'{export_dir}/feature_scaler.model')
        dump(team_encoder, f'{export_dir}/team_encoder.model')

        # Export feature names and other metadata
        metadata = {
            'features': feature_columns,
            'team_names': team_encoder.classes_.tolist()
        }

        with open(f'{export_dir}/metadata_enhanced.json', 'w') as f:
            json.dump(metadata, f)

        print(f'Enhanced models exported successfully to {export_dir}/')
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 