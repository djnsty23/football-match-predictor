import pandas as pd
from datetime import datetime
from predict_enhanced import load_models, predict_match_result
from feature_engineering import load_all_data

def load_historical_matches(start_season=18, end_season=23):
    """Load historical matches from multiple seasons for testing"""
    all_matches = []
    leagues = {
        'england': 'E0',  # Premier League
        'spain': 'SP1',   # La Liga
        'germany': 'D1',  # Bundesliga
        'italy': 'I1',    # Serie A
        'france': 'F1'    # Ligue 1
    }
    
    for country, division in leagues.items():
        for season in range(start_season, end_season + 1):
            try:
                file_path = f'data/{country}-data/{division}_{season:02d}{season+1:02d}.csv'
                season_data = pd.read_csv(file_path)
                season_data['League'] = country
                all_matches.append(season_data)
                print(f"Loaded {country} {division} season {season:02d}-{season+1:02d}")
            except Exception as e:
                print(f"Could not load {country} {division} season {season:02d}-{season+1:02d}: {str(e)}")
    
    if not all_matches:
        raise Exception("No match data could be loaded")
    
    matches_df = pd.concat(all_matches, ignore_index=True)
    
    # Convert date strings to datetime
    def parse_date(date_str):
        date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%d/%m/%y']
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        return pd.NaT
    
    matches_df['Date'] = matches_df['Date'].apply(parse_date)
    
    # Ensure all required columns exist
    required_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 
                       'HS', 'AS', 'HST', 'AST', 'HR', 'AR']
    for col in required_columns:
        if col not in matches_df.columns:
            matches_df[col] = 0
            
    return matches_df.sort_values('Date')

def test_auto_adjustment(num_matches=100):
    """Test the auto-adjustment system on a sequence of historical matches"""
    print("Loading models and data...")
    models = load_models()
    all_data = load_all_data()
    
    try:
        historical_matches = load_historical_matches()
        print(f"Loaded {len(historical_matches)} historical matches")
    except Exception as e:
        print(f"Error loading historical matches: {str(e)}")
        return None
    
    print(f"\nTesting auto-adjustment system on {num_matches} matches...")
    print("=" * 80)
    
    # Track performance metrics
    correct_predictions = 0
    total_predictions = 0
    model_weights_history = []
    confidence_thresholds_history = []
    
    # Get a sample of matches to test
    test_matches = historical_matches.sample(n=min(num_matches, len(historical_matches)), random_state=42)
    test_matches = test_matches.sort_values('Date')
    
    for idx, match in test_matches.iterrows():
        print(f"\nMatch {total_predictions + 1}/{num_matches}")
        print(f"Date: {match['Date'].strftime('%Y-%m-%d')}")
        print(f"{match['HomeTeam']} vs {match['AwayTeam']}")
        
        try:
            # Make prediction
            result = predict_match_result(match, all_data, models)
            
            if result is not None:
                total_predictions += 1
                
                # Extract and store model weights and thresholds
                _, _, _, _, _, _, model_adjuster = models
                
                # Get current weights and thresholds
                current_weights = model_adjuster.history.get('current_weights', {
                    'rf': 0.4,
                    'lr': 0.35,
                    'nb': 0.25
                })
                current_thresholds = model_adjuster.adjust_confidence_thresholds()
                
                model_weights_history.append(current_weights)
                confidence_thresholds_history.append(current_thresholds)
                
                # Check if prediction was correct
                if 'FTR' in match:
                    actual_result = match['FTR']
                    predicted_result = result.get('ensemble_pred')
                    if predicted_result == actual_result:
                        correct_predictions += 1
                
                # Calculate running accuracy
                if total_predictions > 0:
                    running_accuracy = correct_predictions / total_predictions
                    print(f"\nRunning Accuracy: {running_accuracy:.3f}")
        except Exception as e:
            print(f"Error processing match: {str(e)}")
            continue
        
        print("-" * 80)
    
    # Print final performance analysis
    print("\nFinal Performance Analysis")
    print("=" * 80)
    
    if total_predictions > 0:
        final_accuracy = correct_predictions / total_predictions
        print(f"Overall Accuracy: {final_accuracy:.3f}")
        
        # Analyze weight evolution
        print("\nModel Weight Evolution:")
        for model in ['rf', 'lr', 'nb']:
            initial_weight = model_weights_history[0][model]
            final_weight = model_weights_history[-1][model]
            print(f"{model.upper()}: {initial_weight:.3f} -> {final_weight:.3f}")
        
        # Analyze threshold evolution
        print("\nConfidence Threshold Evolution:")
        if confidence_thresholds_history[0] and confidence_thresholds_history[-1]:
            for model in ['rf', 'lr', 'nb', 'ensemble']:
                initial_threshold = confidence_thresholds_history[0][model]
                final_threshold = confidence_thresholds_history[-1][model]
                print(f"{model.upper()}: {initial_threshold:.3f} -> {final_threshold:.3f}")
    
    return {
        'accuracy': final_accuracy if total_predictions > 0 else 0,
        'weights_history': model_weights_history,
        'thresholds_history': confidence_thresholds_history
    }

if __name__ == "__main__":
    test_auto_adjustment() 