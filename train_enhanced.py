from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
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
        
        print("Calculating team statistics...")
        team_stats = calculate_team_stats(data)
        
        print("Encoding teams...")
        encoder = LabelEncoder()
        home_encoded = encoder.fit_transform(data['HomeTeam'])
        home_encoded_mapping = dict(
            zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
        data['home_encoded'] = home_encoded

        encoder = LabelEncoder()
        away_encoded = encoder.fit_transform(data['AwayTeam'])
        away_encoded_mapping = dict(
            zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
        data['away_encoded'] = away_encoded
        
        print("Adding enhanced features...")
        enhanced_features = []
        for idx, match in tqdm(data.iterrows(), total=len(data), desc="Processing matches"):
            features = add_features_to_match(match, team_stats, data)
            enhanced_features.append(features)
        
        enhanced_df = pd.DataFrame(enhanced_features)
        data = pd.concat([data, enhanced_df], axis=1)
        
        # Select features for training
        base_features = [
            'home_encoded', 'away_encoded',
            'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR'
        ]
        
        team_features = [
            'home_win_ratio', 'away_win_ratio',
            'home_team_home_win_ratio', 'away_team_away_win_ratio',
            'home_goals_per_game', 'away_goals_per_game',
            'home_comeback_ratio', 'away_comeback_ratio',
            'home_second_half_goal_ratio', 'away_second_half_goal_ratio',
            'home_clean_sheet_ratio', 'away_clean_sheet_ratio',
            'home_scoring_ratio', 'away_scoring_ratio'
        ]
        
        h2h_features = [
            'h2h_matches_played', 'h2h_home_win_ratio', 'h2h_away_win_ratio',
            'h2h_home_goals_per_game', 'h2h_away_goals_per_game'
        ]
        
        form_features = [
            'home_recent_win_ratio', 'away_recent_win_ratio',
            'home_recent_goals_per_game', 'away_recent_goals_per_game',
            'home_recent_clean_sheet_ratio', 'away_recent_clean_sheet_ratio'
        ]
        
        input_features = base_features + team_features + h2h_features + form_features
        
        print("Preparing training data...")
        data = data.dropna(subset=input_features + ['FTR'])
        
        X = data[input_features]
        y = data['FTR']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        print("\nTraining enhanced models...")
        
        print("\nLogistic Regression one vs All Classifier")
        print("-" * 50)
        lr_classifier = LogisticRegression(multi_class='ovr', max_iter=1000)
        model(lr_classifier, X_train, y_train, X_test, y_test)

        print("\nGaussian Naive Bayes Classifier")
        print("-" * 50)
        nb_classifier = GaussianNB()
        model(nb_classifier, X_train, y_train, X_test, y_test)

        print("\nTuning Random Forest Classifier")
        print("-" * 50)
        rf_classifier = tune_random_forest(X_train, y_train, X_test, y_test)
        
        print("\nExporting enhanced models...")
        exportedModelsPath = 'exportedModels'
        makedirs(exportedModelsPath, exist_ok=True)

        dump(lr_classifier, f'{exportedModelsPath}/lr_classifier_enhanced.model')
        dump(nb_classifier, f'{exportedModelsPath}/nb_classifier_enhanced.model')
        dump(rf_classifier, f'{exportedModelsPath}/rf_classifier_enhanced.model')
        dump(scaler, f'{exportedModelsPath}/feature_scaler.model')

        exportMetaData = {
            'home_teams': home_encoded_mapping,
            'away_teams': away_encoded_mapping,
            'features': input_features
        }

        with open(f'{exportedModelsPath}/metaData_enhanced.json', 'w') as f:
            json.dump(exportMetaData, f)
            
        with open(f'{exportedModelsPath}/team_stats.json', 'w') as f:
            json.dump({team: dict(stats) for team, stats in team_stats.items()}, f)

        print(f'Enhanced models exported successfully to {exportedModelsPath}/')
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 