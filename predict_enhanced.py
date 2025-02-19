from joblib import load
import json
import numpy as np
import pandas as pd
from os import path
import random
from feature_engineering import add_features_to_match, load_all_data

def load_models():
    # Load the enhanced models and metadata
    lr_classifier = load('exportedModels/lr_classifier_enhanced.model')
    nb_classifier = load('exportedModels/nb_classifier_enhanced.model')
    rf_classifier = load('exportedModels/rf_classifier_enhanced.model')
    scaler = load('exportedModels/feature_scaler.model')
    
    with open('exportedModels/metaData_enhanced.json', 'r') as f:
        metadata = json.load(f)
        
    with open('exportedModels/team_stats.json', 'r') as f:
        team_stats = json.load(f)
    
    return lr_classifier, nb_classifier, rf_classifier, scaler, metadata, team_stats

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

def get_ensemble_prediction(lr_pred, nb_pred, rf_pred, lr_proba, nb_proba, rf_proba):
    """
    Get ensemble prediction using dynamic weighted voting and confidence scores.
    """
    # Get dynamic weights based on model confidence
    weights = get_dynamic_weights(lr_proba, nb_proba, rf_proba)
    
    # Calculate weighted probabilities for each outcome
    ensemble_proba = np.zeros(3)
    for i in range(3):
        ensemble_proba[i] = (
            weights['rf'] * rf_proba[i] +
            weights['lr'] * lr_proba[i] +
            weights['nb'] * nb_proba[i]
        )
    
    # Normalize probabilities
    ensemble_proba = ensemble_proba / ensemble_proba.sum()
    
    # Get ensemble prediction (most likely outcome)
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
    lr_classifier, nb_classifier, rf_classifier, scaler, metadata, team_stats = models
    
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
    additional_features = add_features_to_match(match_data, team_stats, all_data)
    
    # Add recent results tracking
    home_recent_results = []
    home_recent_goals = []
    home_recent_conceded = []
    away_recent_results = []
    away_recent_goals = []
    away_recent_conceded = []
    
    # Get last 5 matches for each team
    recent_matches = all_data[
        ((all_data['HomeTeam'] == home_team) | (all_data['AwayTeam'] == home_team) |
         (all_data['HomeTeam'] == away_team) | (all_data['AwayTeam'] == away_team)) &
        (all_data['Date'] < match_data['Date'])
    ].sort_values('Date', ascending=False).head(10)  # Get 10 to ensure we have 5 for each team
    
    # Process home team recent matches
    home_matches = recent_matches[
        (recent_matches['HomeTeam'] == home_team) |
        (recent_matches['AwayTeam'] == home_team)
    ].head(5)
    
    for _, match in home_matches.iterrows():
        if match['HomeTeam'] == home_team:
            result = 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L'
            goals_scored = match['FTHG']
            goals_conceded = match['FTAG']
        else:
            result = 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L'
            goals_scored = match['FTAG']
            goals_conceded = match['FTHG']
        
        home_recent_results.append(result)
        home_recent_goals.append(goals_scored)
        home_recent_conceded.append(goals_conceded)
    
    # Process away team recent matches
    away_matches = recent_matches[
        (recent_matches['HomeTeam'] == away_team) |
        (recent_matches['AwayTeam'] == away_team)
    ].head(5)
    
    for _, match in away_matches.iterrows():
        if match['HomeTeam'] == away_team:
            result = 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L'
            goals_scored = match['FTHG']
            goals_conceded = match['FTAG']
        else:
            result = 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L'
            goals_scored = match['FTAG']
            goals_conceded = match['FTHG']
        
        away_recent_results.append(result)
        away_recent_goals.append(goals_scored)
        away_recent_conceded.append(goals_conceded)
    
    # Add recent form to additional features
    additional_features['home_recent_results'] = home_recent_results
    additional_features['home_recent_goals'] = home_recent_goals
    additional_features['home_recent_conceded'] = home_recent_conceded
    additional_features['away_recent_results'] = away_recent_results
    additional_features['away_recent_goals'] = away_recent_goals
    additional_features['away_recent_conceded'] = away_recent_conceded
    
    # Calculate league positions (if available)
    try:
        league_table = all_data[all_data['Div'] == match_data['Div']].copy()
        league_table['Points'] = league_table.apply(
            lambda x: 3 if x['FTR'] == 'H' else 1 if x['FTR'] == 'D' else 0, axis=1
        )
        team_points = league_table.groupby('HomeTeam')['Points'].sum()
        positions = team_points.rank(ascending=False, method='min')
        home_pos = positions.get(home_team, 0)
        away_pos = positions.get(away_team, 0)
        additional_features['league_position_diff'] = home_pos - away_pos
    except Exception as e:
        print(f"Warning: Could not calculate league positions - {str(e)}")
        additional_features['league_position_diff'] = 0
    
    # Create feature dictionary with proper names
    feature_dict = {
        'home_encoded': home_encoded,
        'away_encoded': away_encoded,
        'HTHG': match_data['HTHG'],
        'HTAG': match_data['HTAG'],
        'HS': match_data['HS'],
        'AS': match_data['AS'],
        'HST': match_data['HST'],
        'AST': match_data['AST'],
        'HR': match_data['HR'],
        'AR': match_data['AR'],
        # Team features
        'home_win_ratio': additional_features['home_win_ratio'],
        'away_win_ratio': additional_features['away_win_ratio'],
        'home_team_home_win_ratio': additional_features['home_team_home_win_ratio'],
        'away_team_away_win_ratio': additional_features['away_team_away_win_ratio'],
        'home_goals_per_game': additional_features['home_goals_per_game'],
        'away_goals_per_game': additional_features['away_goals_per_game'],
        'home_comeback_ratio': additional_features['home_comeback_ratio'],
        'away_comeback_ratio': additional_features['away_comeback_ratio'],
        'home_second_half_goal_ratio': additional_features['home_second_half_goal_ratio'],
        'away_second_half_goal_ratio': additional_features['away_second_half_goal_ratio'],
        'home_clean_sheet_ratio': additional_features['home_clean_sheet_ratio'],
        'away_clean_sheet_ratio': additional_features['away_clean_sheet_ratio'],
        'home_scoring_ratio': additional_features['home_scoring_ratio'],
        'away_scoring_ratio': additional_features['away_scoring_ratio'],
        # Head-to-head features
        'h2h_matches_played': additional_features['h2h_matches_played'],
        'h2h_home_win_ratio': additional_features['h2h_home_win_ratio'],
        'h2h_away_win_ratio': additional_features['h2h_away_win_ratio'],
        'h2h_home_goals_per_game': additional_features['h2h_home_goals_per_game'],
        'h2h_away_goals_per_game': additional_features['h2h_away_goals_per_game'],
        # Form features
        'home_recent_win_ratio': additional_features['home_recent_win_ratio'],
        'away_recent_win_ratio': additional_features['away_recent_win_ratio'],
        'home_recent_goals_per_game': additional_features['home_recent_goals_per_game'],
        'away_recent_goals_per_game': additional_features['away_recent_goals_per_game'],
        'home_recent_clean_sheet_ratio': additional_features['home_recent_clean_sheet_ratio'],
        'away_recent_clean_sheet_ratio': additional_features['away_recent_clean_sheet_ratio']
    }
    
    # Create DataFrame with features in the correct order
    feature_names = metadata['features']
    X = pd.DataFrame([feature_dict], columns=feature_names)
    
    # Replace any NaN values with 0
    X = X.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Get predictions from all models
    lr_pred = lr_classifier.predict(X_scaled)[0]
    nb_pred = nb_classifier.predict(X_scaled)[0]
    rf_pred = rf_classifier.predict(X_scaled)[0]
    
    # Get prediction probabilities
    lr_proba = lr_classifier.predict_proba(X_scaled)[0]
    nb_proba = nb_classifier.predict_proba(X_scaled)[0]
    rf_proba = rf_classifier.predict_proba(X_scaled)[0]
    
    # Get ensemble prediction with dynamic weights
    ensemble_pred, ensemble_proba, weights = get_ensemble_prediction(
        lr_pred, nb_pred, rf_pred,
        lr_proba, nb_proba, rf_proba
    )
    
    # Get team trends
    home_trend = analyze_team_trend(
        recent_results=home_recent_results,
        recent_goals_scored=home_recent_goals,
        recent_goals_conceded=home_recent_conceded
    )
    
    away_trend = analyze_team_trend(
        recent_results=away_recent_results,
        recent_goals_scored=away_recent_goals,
        recent_goals_conceded=away_recent_conceded
    )
    
    # Calculate confidence with historical data penalties
    lr_confidence = get_prediction_confidence(
        lr_proba,
        h2h_matches=additional_features['h2h_matches_played'],
        recent_form_matches=len(home_recent_results)
    )
    nb_confidence = get_prediction_confidence(
        nb_proba,
        h2h_matches=additional_features['h2h_matches_played'],
        recent_form_matches=len(home_recent_results)
    )
    rf_confidence = get_prediction_confidence(
        rf_proba,
        h2h_matches=additional_features['h2h_matches_played'],
        recent_form_matches=len(home_recent_results)
    )
    ensemble_confidence = get_prediction_confidence(
        ensemble_proba,
        h2h_matches=additional_features['h2h_matches_played'],
        recent_form_matches=len(home_recent_results)
    )
    
    # Adjust odds based on trends and other factors
    base_odds = {
        'Home': 200,
        'Draw': 250,
        'Away': 150
    }
    
    adjusted_odds = {
        'Home': adjust_odds_dynamically(
            base_odds['Home'],
            team_trend=home_trend,
            league_position_diff=additional_features['league_position_diff']
        ),
        'Draw': adjust_odds_dynamically(
            base_odds['Draw'],
            team_trend={'trend_strength': 0.5},  # Neutral for draw
            league_position_diff=0
        ),
        'Away': adjust_odds_dynamically(
            base_odds['Away'],
            team_trend=away_trend,
            league_position_diff=-additional_features['league_position_diff']
        )
    }
    
    # Get detailed head-to-head analysis
    h2h_stats = analyze_head_to_head(
        home_team,
        away_team,
        all_data,
        match_data['Date']
    )
    
    # Get betting recommendations with enhanced analysis
    betting_recommendations = get_betting_recommendation(
        ensemble_proba,
        ensemble_confidence,
        bankroll=1000,
        additional_features=additional_features,
        adjusted_odds=adjusted_odds,
        home_trend=home_trend,
        away_trend=away_trend,
        h2h_stats=h2h_stats
    )
    
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
    print(f"- Recent form (wins): {additional_features['home_recent_win_ratio']:.3f}")
    print(f"- Recent goals per game: {additional_features['home_recent_goals_per_game']:.2f}")
    print(f"- Clean sheet ratio: {additional_features['home_clean_sheet_ratio']:.3f}")
    print(f"- Comeback ratio: {additional_features['home_comeback_ratio']:.3f}")
    
    print(f"\n{away_team}:")
    print(f"- Overall win ratio: {additional_features['away_win_ratio']:.3f}")
    print(f"- Away win ratio: {additional_features['away_team_away_win_ratio']:.3f}")
    print(f"- Goals per game: {additional_features['away_goals_per_game']:.2f}")
    print(f"- Recent form (wins): {additional_features['away_recent_win_ratio']:.3f}")
    print(f"- Recent goals per game: {additional_features['away_recent_goals_per_game']:.2f}")
    print(f"- Clean sheet ratio: {additional_features['away_clean_sheet_ratio']:.3f}")
    print(f"- Comeback ratio: {additional_features['away_comeback_ratio']:.3f}")
    
    print("\nHead-to-Head Statistics:")
    print("-" * 50)
    print(f"Previous meetings: {int(additional_features['h2h_matches_played'])}")
    if additional_features['h2h_matches_played'] > 0:
        print(f"{home_team} wins: {additional_features['h2h_home_win_ratio']:.3f}")
        print(f"{away_team} wins: {additional_features['h2h_away_win_ratio']:.3f}")
        print(f"Goals per game: {home_team}: {additional_features['h2h_home_goals_per_game']:.2f}, {away_team}: {additional_features['h2h_away_goals_per_game']:.2f}")
    
    print(f"\nActual full-time result: {match_data['FTR']} (Score: {int(match_data['FTHG'])} - {int(match_data['FTAG'])})")
    
    print("\nModel Predictions:")
    print("-" * 50)
    print(f"Random Forest ({weights['rf']:.1%}): {rf_pred} (H: {rf_proba[0]:.2f}, D: {rf_proba[1]:.2f}, A: {rf_proba[2]:.2f}) - Confidence: {rf_confidence:.2f}")
    print(f"Logistic Regression ({weights['lr']:.1%}): {lr_pred} (H: {lr_proba[0]:.2f}, D: {lr_proba[1]:.2f}, A: {lr_proba[2]:.2f}) - Confidence: {lr_confidence:.2f}")
    print(f"Naive Bayes ({weights['nb']:.1%}): {nb_pred} (H: {nb_proba[0]:.2f}, D: {nb_proba[1]:.2f}, A: {nb_proba[2]:.2f}) - Confidence: {nb_confidence:.2f}")
    
    print("\nEnsemble Prediction:")
    print("-" * 50)
    print(f"Final Prediction: {ensemble_pred} (H: {ensemble_proba[0]:.2f}, D: {ensemble_proba[1]:.2f}, A: {ensemble_proba[2]:.2f})")
    print(f"Overall Confidence: {ensemble_confidence:.2f}")
    
    # Refined confidence thresholds with more detailed recommendations
    if ensemble_confidence > 0.8:
        print("Confidence Level: Very High")
        print("Recommendation: Strong bet opportunity with high certainty")
    elif ensemble_confidence > 0.65:
        print("Confidence Level: High")
        print("Recommendation: Good bet opportunity")
    elif ensemble_confidence > 0.5:
        print("Confidence Level: Moderate")
        print("Recommendation: Consider betting with caution")
    else:
        print("Confidence Level: Low")
        print("Recommendation: High uncertainty, avoid betting")
    
    print("\nTeam Form Trends:")
    print("-" * 50)
    print(f"{home_team}:")
    print(f"- Form trend: {home_trend['form_trend']}")
    print(f"- Scoring trend: {home_trend['scoring_trend']}")
    print(f"- Defensive trend: {home_trend['defensive_trend']}")
    print(f"- Overall trend strength: {home_trend['trend_strength']:.2f}")
    
    print(f"\n{away_team}:")
    print(f"- Form trend: {away_trend['form_trend']}")
    print(f"- Scoring trend: {away_trend['scoring_trend']}")
    print(f"- Defensive trend: {away_trend['defensive_trend']}")
    print(f"- Overall trend strength: {away_trend['trend_strength']:.2f}")
    
    print("\nDetailed Head-to-Head Analysis:")
    print("-" * 50)
    if h2h_stats:
        print(f"Historical Pattern: {h2h_stats['historical_pattern']}")
        print(f"Scoring Pattern: {h2h_stats['scoring_pattern']}")
        print(f"Venue Importance: {h2h_stats['venue_importance']:.2f}")
        print(f"Dominance Score: {h2h_stats['dominance_score']:.2f}")
    
    if betting_recommendations:
        print("\nBetting Analysis:")
        print("-" * 50)
        for rec in betting_recommendations:
            metrics = rec['metrics']
            risk = rec['risk']
            
            print(f"\n{rec['outcome']} - Odds: {rec['odds']:+d}")
            print(f"Prediction Metrics:")
            print(f"- Model probability: {rec['probability']:.2%}")
            print(f"- Implied probability: {metrics['implied_prob']:.2%}")
            print(f"- True odds (fair value): {(1/rec['probability']-1)*100:+.0f}")
            
            print("\nValue Analysis:")
            print(f"- Edge: {metrics['edge']:+.1f}%")
            print(f"- Expected value: {metrics['value']:.2f}")
            print(f"- ROI: {metrics['roi']:+.1f}%")
            print(f"- Risk/Reward ratio: {metrics['risk_reward_ratio']:.2f}")
            
            print("\nStaking Strategy:")
            print(f"- Risk Level: {risk['level']} ({risk['risk_score']:.2f})")
            print(f"- {risk['description']}")
            print(f"- Recommended stake: ${risk['max_stake']:.2f}")
            print(f"- Potential return: ${metrics['potential_returns']['quarter_kelly']:.2f}")
            
            print("\nKelly Criterion Stakes:")
            print(f"- Conservative (1/4): {metrics['quarter_kelly']:.1%}")
            print(f"- Moderate (1/2): {metrics['half_kelly']:.1%}")
            print(f"- Aggressive (full): {metrics['full_kelly']:.1%}")
    else:
        print("\nNo valuable betting opportunities found")

def main():
    print("Loading enhanced models...")
    models = load_models()
    
    print("Loading random match data...")
    match_data, all_data = load_match_data()
    
    if match_data is not None:
        predict_match_result(match_data, all_data, models)

if __name__ == "__main__":
    main() 