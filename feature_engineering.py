import pandas as pd
import numpy as np
from os import path
from collections import defaultdict
from datetime import datetime

def load_all_data():
    """Load data from all leagues and seasons"""
    data_folders = [
        'english-premier-league_zip',
        'spanish-la-liga_zip',
        'french-ligue-1_zip',
        'german-bundesliga_zip',
        'italian-serie-a_zip'
    ]
    
    all_data = []
    for folder in data_folders:
        for season in range(9, 19):  # 2009-2018
            file_path = f'data/{folder}/data/season-{season:02d}{season+1:02d}_csv.csv'
            if path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Season'] = f'{2000+season}-{season+1}'
                
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
                
                df['Date'] = df['Date'].apply(parse_date)
                all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def calculate_head_to_head(data, team1, team2, before_date=None):
    """Calculate head-to-head statistics between two teams"""
    matches = data[
        (((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
         ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1)))
    ]
    
    if before_date:
        matches = matches[matches['Date'] < before_date]
    
    if len(matches) == 0:
        return {
            'matches_played': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'team1_goals': 0,
            'team2_goals': 0
        }
    
    stats = {
        'matches_played': len(matches),
        'team1_wins': 0,
        'team2_wins': 0,
        'draws': 0,
        'team1_goals': 0,
        'team2_goals': 0
    }
    
    for _, match in matches.iterrows():
        if match['HomeTeam'] == team1:
            stats['team1_goals'] += match['FTHG']
            stats['team2_goals'] += match['FTAG']
            if match['FTR'] == 'H':
                stats['team1_wins'] += 1
            elif match['FTR'] == 'A':
                stats['team2_wins'] += 1
            else:
                stats['draws'] += 1
        else:
            stats['team1_goals'] += match['FTAG']
            stats['team2_goals'] += match['FTHG']
            if match['FTR'] == 'A':
                stats['team1_wins'] += 1
            elif match['FTR'] == 'H':
                stats['team2_wins'] += 1
            else:
                stats['draws'] += 1
    
    return stats

def calculate_recent_form(data, team, before_date, num_matches=5):
    """Calculate a team's recent form before a given date"""
    team_matches = data[
        ((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) &
        (data['Date'] < before_date)
    ].sort_values('Date', ascending=False).head(num_matches)
    
    if len(team_matches) == 0:
        return {
            'matches_played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'clean_sheets': 0
        }
    
    stats = {
        'matches_played': len(team_matches),
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'clean_sheets': 0
    }
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team:
            stats['goals_scored'] += match['FTHG']
            stats['goals_conceded'] += match['FTAG']
            if match['FTAG'] == 0:
                stats['clean_sheets'] += 1
            if match['FTR'] == 'H':
                stats['wins'] += 1
            elif match['FTR'] == 'D':
                stats['draws'] += 1
            else:
                stats['losses'] += 1
        else:
            stats['goals_scored'] += match['FTAG']
            stats['goals_conceded'] += match['FTHG']
            if match['FTHG'] == 0:
                stats['clean_sheets'] += 1
            if match['FTR'] == 'A':
                stats['wins'] += 1
            elif match['FTR'] == 'D':
                stats['draws'] += 1
            else:
                stats['losses'] += 1
    
    return stats

def calculate_team_stats(data):
    """Calculate historical performance statistics for each team"""
    team_stats = defaultdict(lambda: {
        'total_matches': 0,
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'home_wins': 0,
        'home_matches': 0,
        'away_wins': 0,
        'away_matches': 0,
        'comebacks_when_behind': 0,
        'times_behind': 0,
        'second_half_goals': 0,
        'second_half_conceded': 0,
        'clean_sheets': 0,
        'failed_to_score': 0
    })
    
    # Process each match
    for _, match in data.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Home team stats
        team_stats[home_team]['total_matches'] += 1
        team_stats[home_team]['home_matches'] += 1
        team_stats[home_team]['goals_scored'] += match['FTHG']
        team_stats[home_team]['goals_conceded'] += match['FTAG']
        team_stats[home_team]['second_half_goals'] += match['FTHG'] - match['HTHG']
        team_stats[home_team]['second_half_conceded'] += match['FTAG'] - match['HTAG']
        
        if match['FTAG'] == 0:
            team_stats[home_team]['clean_sheets'] += 1
        if match['FTHG'] == 0:
            team_stats[home_team]['failed_to_score'] += 1
        
        if match['FTR'] == 'H':
            team_stats[home_team]['wins'] += 1
            team_stats[home_team]['home_wins'] += 1
        elif match['FTR'] == 'D':
            team_stats[home_team]['draws'] += 1
        else:
            team_stats[home_team]['losses'] += 1
            
        if match['HTHG'] < match['HTAG'] and match['FTHG'] >= match['FTAG']:
            team_stats[home_team]['comebacks_when_behind'] += 1
        if match['HTHG'] < match['HTAG']:
            team_stats[home_team]['times_behind'] += 1
        
        # Away team stats
        team_stats[away_team]['total_matches'] += 1
        team_stats[away_team]['away_matches'] += 1
        team_stats[away_team]['goals_scored'] += match['FTAG']
        team_stats[away_team]['goals_conceded'] += match['FTHG']
        team_stats[away_team]['second_half_goals'] += match['FTAG'] - match['HTAG']
        team_stats[away_team]['second_half_conceded'] += match['FTHG'] - match['HTHG']
        
        if match['FTHG'] == 0:
            team_stats[away_team]['clean_sheets'] += 1
        if match['FTAG'] == 0:
            team_stats[away_team]['failed_to_score'] += 1
        
        if match['FTR'] == 'A':
            team_stats[away_team]['wins'] += 1
            team_stats[away_team]['away_wins'] += 1
        elif match['FTR'] == 'D':
            team_stats[away_team]['draws'] += 1
        else:
            team_stats[away_team]['losses'] += 1
            
        if match['HTAG'] < match['HTHG'] and match['FTAG'] >= match['FTHG']:
            team_stats[away_team]['comebacks_when_behind'] += 1
        if match['HTAG'] < match['HTHG']:
            team_stats[away_team]['times_behind'] += 1
    
    # Calculate derived statistics
    for team in team_stats:
        stats = team_stats[team]
        total = stats['total_matches']
        stats['win_ratio'] = stats['wins'] / total if total > 0 else 0
        stats['home_win_ratio'] = stats['home_wins'] / stats['home_matches'] if stats['home_matches'] > 0 else 0
        stats['away_win_ratio'] = stats['away_wins'] / stats['away_matches'] if stats['away_matches'] > 0 else 0
        stats['goals_per_game'] = stats['goals_scored'] / total if total > 0 else 0
        stats['goals_conceded_per_game'] = stats['goals_conceded'] / total if total > 0 else 0
        stats['comeback_ratio'] = stats['comebacks_when_behind'] / stats['times_behind'] if stats['times_behind'] > 0 else 0
        stats['second_half_goal_ratio'] = stats['second_half_goals'] / stats['goals_scored'] if stats['goals_scored'] > 0 else 0
        stats['clean_sheet_ratio'] = stats['clean_sheets'] / total if total > 0 else 0
        stats['scoring_ratio'] = (total - stats['failed_to_score']) / total if total > 0 else 0
    
    return team_stats

def add_features_to_match(match, team_stats, all_data=None):
    """Add engineered features to a match"""
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    features = {
        'home_win_ratio': home_stats['win_ratio'],
        'away_win_ratio': away_stats['win_ratio'],
        'home_team_home_win_ratio': home_stats['home_win_ratio'],
        'away_team_away_win_ratio': away_stats['away_win_ratio'],
        'home_goals_per_game': home_stats['goals_per_game'],
        'away_goals_per_game': away_stats['goals_per_game'],
        'home_comeback_ratio': home_stats['comeback_ratio'],
        'away_comeback_ratio': away_stats['comeback_ratio'],
        'home_second_half_goal_ratio': home_stats['second_half_goal_ratio'],
        'away_second_half_goal_ratio': away_stats['second_half_goal_ratio'],
        'home_clean_sheet_ratio': home_stats['clean_sheet_ratio'],
        'away_clean_sheet_ratio': away_stats['clean_sheet_ratio'],
        'home_scoring_ratio': home_stats['scoring_ratio'],
        'away_scoring_ratio': away_stats['scoring_ratio']
    }
    
    if all_data is not None and 'Date' in match:
        # Add head-to-head features
        h2h_stats = calculate_head_to_head(all_data, home_team, away_team, match['Date'])
        features.update({
            'h2h_matches_played': h2h_stats['matches_played'],
            'h2h_home_win_ratio': h2h_stats['team1_wins'] / h2h_stats['matches_played'] if h2h_stats['matches_played'] > 0 else 0,
            'h2h_away_win_ratio': h2h_stats['team2_wins'] / h2h_stats['matches_played'] if h2h_stats['matches_played'] > 0 else 0,
            'h2h_home_goals_per_game': h2h_stats['team1_goals'] / h2h_stats['matches_played'] if h2h_stats['matches_played'] > 0 else 0,
            'h2h_away_goals_per_game': h2h_stats['team2_goals'] / h2h_stats['matches_played'] if h2h_stats['matches_played'] > 0 else 0
        })
        
        # Add recent form features
        home_form = calculate_recent_form(all_data, home_team, match['Date'])
        away_form = calculate_recent_form(all_data, away_team, match['Date'])
        
        features.update({
            'home_recent_win_ratio': home_form['wins'] / home_form['matches_played'] if home_form['matches_played'] > 0 else 0,
            'away_recent_win_ratio': away_form['wins'] / away_form['matches_played'] if away_form['matches_played'] > 0 else 0,
            'home_recent_goals_per_game': home_form['goals_scored'] / home_form['matches_played'] if home_form['matches_played'] > 0 else 0,
            'away_recent_goals_per_game': away_form['goals_scored'] / away_form['matches_played'] if away_form['matches_played'] > 0 else 0,
            'home_recent_clean_sheet_ratio': home_form['clean_sheets'] / home_form['matches_played'] if home_form['matches_played'] > 0 else 0,
            'away_recent_clean_sheet_ratio': away_form['clean_sheets'] / away_form['matches_played'] if away_form['matches_played'] > 0 else 0
        })
    
    return features

def main():
    print("Loading all match data...")
    data = load_all_data()
    
    print("Calculating team statistics...")
    team_stats = calculate_team_stats(data)
    
    # Save team stats for use in prediction
    import json
    with open('exportedModels/team_stats.json', 'w') as f:
        json.dump({team: dict(stats) for team, stats in team_stats.items()}, f)
    
    print("Team statistics have been calculated and saved.")
    
    # Print some example stats
    print("\nExample team statistics:")
    print("-" * 50)
    example_team = list(team_stats.keys())[0]
    print(f"Stats for {example_team}:")
    for stat, value in team_stats[example_team].items():
        print(f"{stat}: {value:.3f}" if isinstance(value, float) else f"{stat}: {value}")

if __name__ == "__main__":
    main() 