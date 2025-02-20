from joblib import load
import json
import numpy as np
import pandas as pd
from os import path
import random
from feature_engineering import add_features_to_match, load_all_data
from model_adjustment import ModelAdjuster

def load_models():
    """Load all required models and metadata"""
    try:
        # Load the models
        lr_classifier = load('exportedModels/lr_classifier_enhanced.model')
        nb_classifier = load('exportedModels/nb_classifier_enhanced.model')
        rf_classifier = load('exportedModels/rf_classifier_enhanced.model')
        scaler = load('exportedModels/feature_scaler.model')
        
        # Load metadata
        with open('exportedModels/metadata_enhanced.json', 'r') as f:
            metadata = json.load(f)
            
        # Load team stats if available
        team_stats = {}
        if path.exists('exportedModels/team_stats.json'):
            with open('exportedModels/team_stats.json', 'r') as f:
                team_stats = json.load(f)
        
        # Initialize model adjuster
        model_adjuster = ModelAdjuster()
        
        return lr_classifier, nb_classifier, rf_classifier, scaler, metadata, team_stats, model_adjuster
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def load_match_data():
    # Load all data for head-to-head and form calculations
    all_data = load_all_data()
    
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
        return None, None
        
    data = pd.read_csv(data_file)
    
    # Handle date parsing with multiple formats
    def parse_date(date_str):
        date_formats = [
            '%Y-%m-%d',  # YYYY-MM-DD
            '%d/%m/%Y',  # DD/MM/YYYY
            '%d/%m/%y',  # DD/MM/YY
            '%Y/%m/%d',  # YYYY/MM/DD
        ]
        
        for date_format in date_formats:
            try:
                return pd.to_datetime(date_str, format=date_format)
            except ValueError:
                continue
        
        # If none of the explicit formats work, try pandas' default parser
        try:
            return pd.to_datetime(date_str)
        except ValueError:
            print(f"Warning: Could not parse date '{date_str}'")
            return pd.NaT
    
    data['Date'] = data['Date'].apply(parse_date)
    
    # Select a random match from the dataset
    match = data.sample(n=1).iloc[0]
    
    return match, all_data

def get_dynamic_weights(lr_proba, nb_proba, rf_proba):
    """Calculate dynamic weights based on prediction confidence"""
    lr_conf = max(lr_proba)
    nb_conf = max(nb_proba)
    rf_conf = max(rf_proba)
    
    # Base weights
    weights = {
        'rf': 0.5,  # Random Forest base weight
        'lr': 0.3,  # Logistic Regression base weight
        'nb': 0.2   # Naive Bayes base weight
    }
    
    # Adjust weights based on relative confidence
    total_conf = lr_conf + nb_conf + rf_conf
    if total_conf > 0:
        weights['rf'] = 0.4 + (0.2 * rf_conf / total_conf)
        weights['lr'] = 0.3 + (0.1 * lr_conf / total_conf)
        weights['nb'] = 0.3 + (0.1 * nb_conf / total_conf)
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    
    return weights

def calculate_value_bet(odds, prob):
    """Calculate betting value using implied probability"""
    if odds <= 0:  # American odds (negative)
        implied_prob = abs(odds) / (abs(odds) + 100)
    else:  # American odds (positive)
        implied_prob = 100 / (odds + 100)
    
    value = (prob * (1 + odds/100)) - 1 if odds > 0 else (prob * (1 + 100/abs(odds))) - 1
    return value, implied_prob

def get_ensemble_prediction(lr_pred, nb_pred, rf_pred, lr_proba, nb_proba, rf_proba, model_adjuster=None):
    """Get ensemble prediction with dynamic weights"""
    # Default weights if no adjuster provided
    weights = {
        'lr': 0.35,
        'nb': 0.25,
        'rf': 0.40
    }
    
    # Get adjusted weights if available
    if model_adjuster:
        weights = model_adjuster.adjust_model_weights(weights)
    
    # Ensure probabilities are 1D arrays of length 3
    def process_proba(proba, model_name):
        proba = np.array(proba)
        if len(proba.shape) == 2:
            if proba.shape[1] == 3:  # Shape is (n, 3)
                proba = proba[0]  # Take first row since we only have one sample
            elif proba.shape[0] == 3:  # Shape is (3, n)
                proba = proba[:, 0]  # Take first column
        elif len(proba.shape) == 1:
            if proba.shape[0] != 3:
                print(f"Warning: Invalid {model_name} probability shape {proba.shape}, using default")
                proba = np.array([1/3, 1/3, 1/3])
        else:
            print(f"Warning: Invalid {model_name} probability shape {proba.shape}, using default")
            proba = np.array([1/3, 1/3, 1/3])
        
        # Normalize probabilities
        sum_proba = np.sum(proba)
        if sum_proba > 0:
            proba = proba / sum_proba
        else:
            proba = np.array([1/3, 1/3, 1/3])
        
        # Ensure the array is 2D with shape (3,1) for broadcasting
        return proba.reshape(3, 1)
    
    # Process probabilities
    lr_proba = process_proba(lr_proba, 'LR')
    nb_proba = process_proba(nb_proba, 'NB')
    rf_proba = process_proba(rf_proba, 'RF')
    
    # Calculate weighted probabilities
    ensemble_proba = np.zeros((3, 1))  # Initialize array for weighted probabilities
    ensemble_proba += weights['lr'] * lr_proba
    ensemble_proba += weights['nb'] * nb_proba
    ensemble_proba += weights['rf'] * rf_proba
    
    # Convert back to 1D array
    ensemble_proba = ensemble_proba.flatten()
    
    # Normalize ensemble probabilities
    sum_proba = np.sum(ensemble_proba)
    if sum_proba > 0:
        ensemble_proba = ensemble_proba / sum_proba
    else:
        ensemble_proba = np.array([1/3, 1/3, 1/3])
    
    # Get prediction (index of highest probability)
    outcomes = ['H', 'D', 'A']
    ensemble_pred = outcomes[np.argmax(ensemble_proba)]
    
    return ensemble_pred, ensemble_proba, weights

def get_prediction_confidence(proba, h2h_matches=0, recent_form_matches=0):
    """Calculate prediction confidence based on probability distribution and historical data"""
    # Base confidence calculation
    confidence = (max(proba) - np.mean(proba)) / np.std(proba) if np.std(proba) > 0 else 0
    
    # Additional factors for confidence calculation
    max_prob = max(proba)
    remaining_sum = sum(proba) - max_prob
    prob_ratio = max_prob / remaining_sum if remaining_sum > 0 else 1.0
    
    # Calculate entropy (avoiding log(0))
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in proba)
    
    # Historical data penalties
    h2h_penalty = 1.0
    if h2h_matches < 3:
        h2h_penalty = 0.8  # Significant penalty for very limited h2h data
    elif h2h_matches < 5:
        h2h_penalty = 0.9  # Moderate penalty for limited h2h data
        
    form_penalty = 1.0
    if recent_form_matches < 3:
        form_penalty = 0.85  # Significant penalty for very limited form data
    elif recent_form_matches < 5:
        form_penalty = 0.95  # Moderate penalty for limited form data
    
    # Combine factors (weighted average)
    combined_confidence = (
        0.35 * confidence +
        0.35 * min(prob_ratio/3, 1.0) +
        0.15 * (1 - entropy/1.58) +
        0.15 * min(h2h_penalty * form_penalty, 1.0)
    )
    
    return min(combined_confidence, 1.0)  # Normalize to [0, 1]

def calculate_bet_metrics(prob, odds, bankroll=1000):
    """Calculate comprehensive betting metrics"""
    try:
        if not isinstance(odds, (int, float)):
            raise ValueError("Invalid odds format")
            
        if odds == 0:
            raise ValueError("Odds cannot be zero")
            
        if prob <= 0 or prob > 1:
            raise ValueError("Probability must be between 0 and 1")
            
        if bankroll <= 0:
            raise ValueError("Bankroll must be positive")
            
        if odds <= 0:  # American odds (negative)
            implied_prob = abs(odds) / (abs(odds) + 100)
            decimal_odds = 1 + (100 / abs(odds))
        else:  # American odds (positive)
            implied_prob = 100 / (odds + 100)
            decimal_odds = 1 + (odds / 100)
        
        # Basic metrics
        value = (prob * decimal_odds) - 1
        edge = (prob - implied_prob) * 100
        
        # Kelly criterion and variants
        full_kelly = max(0, (prob * decimal_odds - 1) / (decimal_odds - 1))
        half_kelly = full_kelly / 2  # Conservative Kelly
        quarter_kelly = full_kelly / 4  # Very conservative Kelly
        
        # Risk-adjusted metrics
        roi = (value / implied_prob) * 100 if implied_prob > 0 else 0  # Return on Investment
        risk_reward_ratio = edge / (100 - prob * 100) if prob < 1 else 0  # Risk-Reward Ratio
        
        # Potential returns
        stake_sizes = {
            'quarter_kelly': quarter_kelly * bankroll,
            'half_kelly': half_kelly * bankroll,
            'full_kelly': full_kelly * bankroll
        }
        
        potential_returns = {
            size: (stake * (decimal_odds - 1)) 
            for size, stake in stake_sizes.items()
        }
        
        return {
            'implied_prob': implied_prob,
            'decimal_odds': decimal_odds,
            'value': value,
            'edge': edge,
            'full_kelly': full_kelly,
            'half_kelly': half_kelly,
            'quarter_kelly': quarter_kelly,
            'roi': roi,
            'risk_reward_ratio': risk_reward_ratio,
            'stake_sizes': stake_sizes,
            'potential_returns': potential_returns
        }
    except Exception as e:
        print(f"Warning: Error calculating betting metrics - {str(e)}")
        return None

def analyze_team_trend(recent_results, recent_goals_scored, recent_goals_conceded):
    """Analyze team's recent performance trend with weighted recent results"""
    if not recent_results or len(recent_results) < 3:
        return {
            'form_trend': 'Unknown',
            'scoring_trend': 'Unknown',
            'defensive_trend': 'Unknown',
            'trend_strength': 0.0,
            'momentum_score': 0.0,
            'consistency_score': 0.0
        }
    
    # Exponential weights for recency (most recent matches have higher weight)
    weights = np.exp([-(i/2) for i in range(len(recent_results))])
    weights = weights / weights.sum()
    
    # Analyze form trend with weighted results
    result_values = [3 if res == 'W' else 1 if res == 'D' else 0 for res in recent_results]
    weighted_form = sum(w * v for w, v in zip(weights, result_values))
    form_trend = weighted_form / 3  # Normalize to [0, 1]
    
    # Analyze scoring trend with weighted goals
    weighted_scoring = sum(w * g for w, g in zip(weights, recent_goals_scored))
    avg_scoring = np.mean(recent_goals_scored)
    scoring_trend = weighted_scoring / (avg_scoring + 1)  # Normalize and avoid division by zero
    
    # Analyze defensive trend with weighted goals conceded
    weighted_defensive = sum(w * g for w, g in zip(weights, recent_goals_conceded))
    avg_defensive = np.mean(recent_goals_conceded)
    defensive_trend = 1 - (weighted_defensive / (avg_defensive + 1))  # Inverse for defense
    
    # Calculate momentum (recent performance vs overall performance)
    recent_points = sum(result_values[:3]) / 9  # Last 3 matches
    overall_points = sum(result_values) / (3 * len(result_values))
    momentum_score = max(0, min(1, recent_points / (overall_points + 0.1)))
    
    # Calculate consistency
    consistency_score = 1 - np.std(result_values) / 3  # Lower variance means higher consistency
    
    # Calculate overall trend strength with weighted components
    trend_strength = (
        0.3 * form_trend +
        0.2 * scoring_trend +
        0.2 * defensive_trend +
        0.15 * momentum_score +
        0.15 * consistency_score
    )
    
    return {
        'form_trend': 'Improving' if form_trend > 0.6 else 'Declining' if form_trend < 0.4 else 'Stable',
        'scoring_trend': 'Improving' if scoring_trend > 1.2 else 'Declining' if scoring_trend < 0.8 else 'Stable',
        'defensive_trend': 'Improving' if defensive_trend > 0.6 else 'Declining' if defensive_trend < 0.4 else 'Stable',
        'trend_strength': trend_strength,
        'momentum_score': momentum_score,
        'consistency_score': consistency_score
    }

def analyze_head_to_head(home_team, away_team, all_matches, current_date, num_years=3):
    """Detailed head-to-head analysis with historical patterns"""
    try:
        # Filter matches between these teams
        h2h_matches = all_matches[
            ((all_matches['HomeTeam'] == home_team) & (all_matches['AwayTeam'] == away_team)) |
            ((all_matches['HomeTeam'] == away_team) & (all_matches['AwayTeam'] == home_team))
        ].copy()
        
        if len(h2h_matches) == 0:
            return {
                'matches_played': 0,
                'home_wins': 0,
                'away_wins': 0,
                'draws': 0,
                'home_goals_avg': 0,
                'away_goals_avg': 0,
                'dominance_score': 0.5,
                'historical_pattern': 'No history',
                'venue_importance': 0.5,
                'scoring_pattern': 'Unknown'
            }
        
        # Calculate basic stats
        stats = {
            'matches_played': len(h2h_matches),
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'home_goals': [],
            'away_goals': [],
            'home_wins_at_venue': 0,
            'matches_at_venue': 0
        }
        
        # Analyze each match
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team:
                stats['home_goals'].append(match['FTHG'])
                stats['away_goals'].append(match['FTAG'])
                stats['matches_at_venue'] += 1
                if match['FTR'] == 'H':
                    stats['home_wins'] += 1
                    stats['home_wins_at_venue'] += 1
                elif match['FTR'] == 'A':
                    stats['away_wins'] += 1
                else:
                    stats['draws'] += 1
            else:
                stats['home_goals'].append(match['FTAG'])
                stats['away_goals'].append(match['FTHG'])
                if match['FTR'] == 'A':
                    stats['home_wins'] += 1
                elif match['FTR'] == 'H':
                    stats['away_wins'] += 1
                else:
                    stats['draws'] += 1
        
        # Calculate advanced metrics
        stats['home_goals_avg'] = np.mean(stats['home_goals'])
        stats['away_goals_avg'] = np.mean(stats['away_goals'])
        
        # Calculate dominance score (weighted towards recent matches)
        total_matches = stats['matches_played']
        home_win_ratio = stats['home_wins'] / total_matches
        away_win_ratio = stats['away_wins'] / total_matches
        stats['dominance_score'] = (home_win_ratio / (home_win_ratio + away_win_ratio)) if (home_win_ratio + away_win_ratio) > 0 else 0.5
        
        # Analyze venue importance
        stats['venue_importance'] = stats['home_wins_at_venue'] / stats['matches_at_venue'] if stats['matches_at_venue'] > 0 else 0.5
        
        # Determine historical pattern
        if stats['home_wins'] > 2 * stats['away_wins']:
            stats['historical_pattern'] = 'Strong home team dominance'
        elif stats['away_wins'] > 2 * stats['home_wins']:
            stats['historical_pattern'] = 'Strong away team dominance'
        elif abs(stats['home_wins'] - stats['away_wins']) <= 1:
            stats['historical_pattern'] = 'Evenly matched'
        else:
            stats['historical_pattern'] = 'Slight advantage to ' + (home_team if stats['home_wins'] > stats['away_wins'] else away_team)
        
        # Analyze scoring pattern
        avg_total_goals = (stats['home_goals_avg'] + stats['away_goals_avg'])
        if avg_total_goals > 3.5:
            stats['scoring_pattern'] = 'High scoring matches'
        elif avg_total_goals < 2:
            stats['scoring_pattern'] = 'Low scoring matches'
        else:
            stats['scoring_pattern'] = 'Moderate scoring matches'
        
        return stats
    except Exception as e:
        print(f"Warning: Error in head-to-head analysis - {str(e)}")
        return None

def adjust_odds_dynamically(base_odds, market_movement=0, team_trend=None, league_position_diff=0):
    """Adjust odds based on various dynamic factors"""
    try:
        if not isinstance(base_odds, (int, float)) or base_odds == 0:
            return base_odds
            
        # Convert American odds to decimal for easier manipulation
        if base_odds > 0:
            decimal_odds = 1 + (base_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(base_odds))
        
        # Apply market movement adjustment (typically -5% to +5%)
        market_factor = 1 + (market_movement / 100)
        decimal_odds *= market_factor
        
        # Apply team trend adjustment
        if team_trend and team_trend['trend_strength'] > 0:
            trend_factor = 1 + ((team_trend['trend_strength'] - 0.5) * 0.1)  # Max ±5% adjustment
            decimal_odds *= trend_factor
        
        # Apply league position adjustment
        position_factor = 1 + (league_position_diff * 0.01)  # 1% per position difference
        decimal_odds *= position_factor
        
        # Convert back to American odds
        if decimal_odds > 2:
            american_odds = (decimal_odds - 1) * 100
        else:
            american_odds = -100 / (decimal_odds - 1)
        
        return int(round(american_odds))
    except Exception as e:
        print(f"Warning: Error adjusting odds - {str(e)}")
        return base_odds

def assess_bet_risk(metrics, confidence, trend_strength=0.5, h2h_stats=None):
    """Enhanced risk assessment with more factors and fine-tuned parameters"""
    try:
        # Base risk factors
        risk_factors = {
            'edge_risk': 1 - min(metrics['edge'] / 15, 1),  # Edge below 15% increases risk
            'kelly_risk': metrics['full_kelly'],  # Higher Kelly fraction means higher risk
            'confidence_risk': 1 - confidence,  # Lower confidence means higher risk
            'value_risk': 1 - min(metrics['value'] / 0.5, 1),  # Value below 0.5 increases risk
            'roi_risk': 1 - min(metrics['roi'] / 30, 1),  # ROI below 30% increases risk
            'trend_risk': 1 - trend_strength  # Poor trend increases risk
        }
        
        # Add head-to-head factors if available
        if h2h_stats:
            # Historical consistency risk
            matches_played = h2h_stats['matches_played']
            if matches_played > 0:
                historical_risk = 1 - (matches_played / 10)  # More matches means lower risk
                dominance_factor = abs(h2h_stats['dominance_score'] - 0.5) * 2  # Clear dominance reduces risk
                venue_factor = abs(h2h_stats['venue_importance'] - 0.5) * 2  # Clear venue advantage reduces risk
                
                risk_factors['historical_risk'] = max(0, min(1, historical_risk - dominance_factor - venue_factor))
            else:
                risk_factors['historical_risk'] = 0.8  # High risk for no history
        
        # Weight the risk factors
        weights = {
            'edge_risk': 0.25,
            'kelly_risk': 0.20,
            'confidence_risk': 0.20,
            'value_risk': 0.15,
            'roi_risk': 0.10,
            'trend_risk': 0.10
        }
        
        if 'historical_risk' in risk_factors:
            # Redistribute weights to include historical risk
            for k in weights:
                weights[k] *= 0.8  # Reduce other weights by 20%
            weights['historical_risk'] = 0.20  # Add 20% weight for historical risk
        
        weighted_risk = sum(risk_factors[k] * weights[k] for k in risk_factors)
        
        # More granular risk levels with adjusted thresholds
        if weighted_risk < 0.15:
            return {
                'level': 'Minimal',
                'description': 'Very safe bet opportunity',
                'max_stake': metrics['half_kelly'],
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        elif weighted_risk < 0.25:
            return {
                'level': 'Very Low',
                'description': 'Highly conservative bet opportunity',
                'max_stake': metrics['half_kelly'] * 0.8,
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        elif weighted_risk < 0.35:
            return {
                'level': 'Low',
                'description': 'Conservative bet opportunity',
                'max_stake': metrics['quarter_kelly'],
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        elif weighted_risk < 0.45:
            return {
                'level': 'Low-Moderate',
                'description': 'Moderately conservative bet opportunity',
                'max_stake': metrics['quarter_kelly'] * 0.8,
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        elif weighted_risk < 0.55:
            return {
                'level': 'Moderate',
                'description': 'Balanced bet opportunity',
                'max_stake': metrics['quarter_kelly'] * 0.6,
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        elif weighted_risk < 0.65:
            return {
                'level': 'Moderate-High',
                'description': 'Speculative bet opportunity',
                'max_stake': metrics['quarter_kelly'] * 0.4,
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        elif weighted_risk < 0.75:
            return {
                'level': 'High',
                'description': 'High-risk bet opportunity',
                'max_stake': metrics['quarter_kelly'] * 0.3,
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
        else:
            return {
                'level': 'Very High',
                'description': 'Extremely risky bet - Consider avoiding',
                'max_stake': metrics['quarter_kelly'] * 0.2,
                'risk_score': weighted_risk,
                'risk_factors': risk_factors
            }
    except Exception as e:
        print(f"Warning: Error in risk assessment - {str(e)}")
        return None

def get_betting_recommendation(ensemble_proba, confidence, bankroll=1000, additional_features=None, adjusted_odds=None, home_trend=None, away_trend=None, h2h_stats=None):
    """Generate detailed betting recommendations"""
    try:
        if not isinstance(bankroll, (int, float)) or bankroll <= 0:
            raise ValueError("Invalid bankroll amount")
            
        outcomes = ['Home', 'Draw', 'Away']
        
        # Get real-time odds if available, otherwise use placeholder odds
        try:
            if adjusted_odds:
                odds = adjusted_odds
            else:
                odds = {
                    'Home': 200,  # Example odds (can be replaced with real bookmaker odds)
                    'Draw': 250,
                    'Away': 150
                }
        except Exception as e:
            print(f"Warning: Could not fetch real odds - using placeholder odds")
            odds = {
                'Home': 200,
                'Draw': 250,
                'Away': 150
            }
        
        recommendations = []
        for outcome, prob, odd in zip(outcomes, ensemble_proba, odds.values()):
            try:
                if prob <= 0:  # Skip outcomes with zero probability
                    continue
                    
                # Calculate comprehensive betting metrics
                metrics = calculate_bet_metrics(prob, odd, bankroll)
                
                if metrics is None:  # Skip if metrics calculation failed
                    continue
                    
                if metrics['value'] > 0:
                    # Get trend strength for risk assessment
                    trend_strength = 0.5  # Default neutral value
                    if outcome == 'Home' and home_trend:
                        trend_strength = home_trend['trend_strength']
                    elif outcome == 'Away' and away_trend:
                        trend_strength = away_trend['trend_strength']
                    
                    # Assess risk and get betting strategy
                    risk_assessment = assess_bet_risk(metrics, confidence, trend_strength, h2h_stats)
                    
                    recommendations.append({
                        'outcome': outcome,
                        'probability': prob,
                        'odds': odd,
                        'metrics': metrics,
                        'risk': risk_assessment
                    })
            except Exception as e:
                print(f"Warning: Could not process betting recommendation for {outcome} - {str(e)}")
                continue
        
        return sorted(recommendations, key=lambda x: x['metrics']['value'], reverse=True)
    except Exception as e:
        print(f"Error generating betting recommendations: {str(e)}")
        return []

def predict_match_result(match_data, all_data, models):
    lr_classifier, nb_classifier, rf_classifier, scaler, metadata, team_stats, model_adjuster = models
    
    home_team = match_data['HomeTeam']
    away_team = match_data['AwayTeam']
    
    # Get team encodings from metadata
    team_names = metadata.get('team_names', [])
    
    if not team_names:
        print("Error: No team names found in metadata")
        return None
        
    if home_team not in team_names or away_team not in team_names:
        print("Warning: Team(s) not found in training data")
        print(f"Home team '{home_team}' in training data: {home_team in team_names}")
        print(f"Away team '{away_team}' in training data: {away_team in team_names}")
        return None
    
    # Get team encodings
    try:
        home_encoded = team_names.index(home_team)
        away_encoded = team_names.index(away_team)
    except ValueError as e:
        print(f"Error encoding teams: {str(e)}")
        return None
    
    # Initialize default values for required fields
    for field in ['HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR']:
        if field not in match_data or pd.isna(match_data[field]):
            match_data[field] = 0
    
    # Get additional features
    additional_features = add_features_to_match(match_data, team_stats, all_data)
    
    # Create feature dictionary with all features initialized to 0
    feature_dict = {feature: 0.0 for feature in metadata['features']}
    
    # Update with basic match features
    basic_features = {
        'home_encoded': home_encoded,
        'away_encoded': away_encoded,
        'HTHG': float(match_data['HTHG']),
        'HTAG': float(match_data['HTAG']),
        'HS': float(match_data['HS']),
        'AS': float(match_data['AS']),
        'HST': float(match_data['HST']),
        'AST': float(match_data['AST']),
        'HR': float(match_data['HR']),
        'AR': float(match_data['AR'])
    }
    feature_dict.update(basic_features)
    
    # Add additional features that exist in metadata['features']
    for key, value in additional_features.items():
        if key in metadata['features']:
            feature_dict[key] = float(value) if pd.notnull(value) else 0.0
    
    # Create DataFrame with features in the correct order
    X = pd.DataFrame([feature_dict])
    
    # Ensure we only use features that were used in training
    X = X[metadata['features']]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions from each model
    lr_pred = lr_classifier.predict(X_scaled)[0]
    nb_pred = nb_classifier.predict(X_scaled)[0]
    rf_pred = rf_classifier.predict(X_scaled)[0]
    
    # Get raw probabilities
    lr_proba = lr_classifier.predict_proba(X_scaled)
    nb_proba = nb_classifier.predict_proba(X_scaled)
    rf_proba = rf_classifier.predict_proba(X_scaled)
    
    print("Raw probability shapes:")
    print(f"LR: {lr_proba.shape}")
    print(f"NB: {nb_proba.shape}")
    print(f"RF: {rf_proba.shape}")
    
    # Get ensemble prediction with dynamic weights
    ensemble_pred, ensemble_proba, weights = get_ensemble_prediction(
        lr_pred, nb_pred, rf_pred,
        lr_proba, nb_proba, rf_proba,
        model_adjuster
    )
    
    # Calculate confidence scores
    lr_confidence = get_prediction_confidence(lr_proba[0])
    nb_confidence = get_prediction_confidence(nb_proba[0])
    rf_confidence = get_prediction_confidence(rf_proba[0])
    ensemble_confidence = get_prediction_confidence(ensemble_proba)
    
    # Record prediction for future adjustments if actual result is available
    if 'FTR' in match_data:
        predictions = {
            'lr_pred': lr_pred,
            'nb_pred': nb_pred,
            'rf_pred': rf_pred,
            'ensemble_pred': ensemble_pred,
            'lr_proba': lr_proba[0],
            'nb_proba': nb_proba[0],
            'rf_proba': rf_proba[0],
            'ensemble_proba': ensemble_proba,
            'lr_confidence': lr_confidence,
            'nb_confidence': nb_confidence,
            'rf_confidence': rf_confidence,
            'ensemble_confidence': ensemble_confidence
        }
    
    return {
        'ensemble_pred': ensemble_pred,
        'ensemble_proba': ensemble_proba.tolist(),
        'ensemble_confidence': ensemble_confidence,
        'model_weights': weights,
        'individual_predictions': {
            'lr': {'pred': lr_pred, 'proba': lr_proba[0].tolist(), 'confidence': lr_confidence},
            'nb': {'pred': nb_pred, 'proba': nb_proba[0].tolist(), 'confidence': nb_confidence},
            'rf': {'pred': rf_pred, 'proba': rf_proba[0].tolist(), 'confidence': rf_confidence}
        }
    }

def main():
    print("Loading enhanced models...")
    models = load_models()
    
    print("Loading random match data...")
    match_data, all_data = load_match_data()
    
    if match_data is not None:
        predict_match_result(match_data, all_data, models)

if __name__ == "__main__":
    main() 