import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # Suppress pandas warnings

def load_league_data(country, division, seasons=None, encoding='utf-8'):
    """Load and combine data from multiple seasons for a specific league"""
    data_dir = Path('data') / f"{country}-data"
    all_data = []
    
    # If no seasons specified, load all available
    if seasons is None:
        files = list(data_dir.glob(f"{division}_*.csv"))
        seasons = [f.stem.split('_')[1] for f in files]
    
    for season in sorted(seasons):
        file_path = data_dir / f"{division}_{season}.csv"
        if file_path.exists():
            try:
                # Try different encodings if utf-8 fails
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
                
                df['Season'] = season
                df['Season_Start'] = int('20' + season[:2])  # Convert season code to year
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        return None
    
    combined_data = pd.concat(all_data, ignore_index=True)
    return clean_and_standardize_data(combined_data)

def clean_and_standardize_data(df):
    """Clean and standardize the dataset"""
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Convert date strings to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
    
    # Ensure essential columns exist and have correct types
    essential_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
    for col in essential_columns:
        if col not in df.columns:
            print(f"Warning: Missing essential column {col}")
            return None
    
    # Convert score columns to int
    score_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Standardize result columns
    df['FTR'] = df['FTR'].map({'H': 'H', 'D': 'D', 'A': 'A'})
    df['HTR'] = df['HTR'].map({'H': 'H', 'D': 'D', 'A': 'A'})
    
    return df

def analyze_league_trends(df, league_name):
    """Analyze trends in league data"""
    print(f"\nAnalyzing trends for {league_name}...")
    
    # Create season summary
    season_summary = df.groupby('Season').agg({
        'FTHG': ['mean', 'sum'],
        'FTAG': ['mean', 'sum'],
        'FTR': lambda x: (x == 'H').mean(),  # Home win ratio
    }).round(3)
    
    print("\nSeason Summary:")
    print(season_summary)
    
    # Calculate overall statistics
    total_matches = len(df)
    home_wins = (df['FTR'] == 'H').sum()
    away_wins = (df['FTR'] == 'A').sum()
    draws = (df['FTR'] == 'D').sum()
    
    print(f"\nOverall Statistics ({total_matches} matches):")
    print(f"Home Wins: {home_wins} ({home_wins/total_matches:.1%})")
    print(f"Away Wins: {away_wins} ({away_wins/total_matches:.1%})")
    print(f"Draws: {draws} ({draws/total_matches:.1%})")
    
    # Analyze goal patterns
    print("\nGoal Patterns:")
    print(f"Average Home Goals: {df['FTHG'].mean():.2f}")
    print(f"Average Away Goals: {df['FTAG'].mean():.2f}")
    
    # Analyze second half comebacks
    comebacks = df[
        ((df['HTR'] == 'H') & (df['FTR'] == 'A')) |
        ((df['HTR'] == 'A') & (df['FTR'] == 'H'))
    ]
    print(f"\nComebacks: {len(comebacks)} ({len(comebacks)/total_matches:.1%} of matches)")
    
    # Analyze team performance
    if len(df) > 0:
        print("\nTop Performing Teams:")
        team_stats = pd.DataFrame()
        
        # Home performance
        home_stats = df.groupby('HomeTeam').agg({
            'FTHG': 'mean',
            'FTR': lambda x: (x == 'H').mean()
        }).round(3)
        home_stats.columns = ['Avg_Home_Goals', 'Home_Win_Rate']
        
        # Away performance
        away_stats = df.groupby('AwayTeam').agg({
            'FTAG': 'mean',
            'FTR': lambda x: (x == 'A').mean()
        }).round(3)
        away_stats.columns = ['Avg_Away_Goals', 'Away_Win_Rate']
        
        # Combine stats
        team_stats = home_stats.join(away_stats)
        team_stats['Overall_Rating'] = (
            team_stats['Home_Win_Rate'] * 0.4 +
            team_stats['Away_Win_Rate'] * 0.4 +
            team_stats['Avg_Home_Goals'] * 0.1 +
            team_stats['Avg_Away_Goals'] * 0.1
        )
        
        print("\nTop 5 Teams by Overall Performance:")
        print(team_stats.nlargest(5, 'Overall_Rating'))
    
    return season_summary

def plot_league_trends(df, league_name, output_dir='analysis_plots'):
    """Create visualizations of league trends"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')  # Use default style instead of seaborn
    sns.set_theme()  # Apply seaborn theme
    
    # 1. Goals per season trend
    plt.figure(figsize=(12, 6))
    season_goals = df.groupby('Season').agg({
        'FTHG': 'mean',
        'FTAG': 'mean'
    })
    season_goals.plot(title=f'{league_name} - Average Goals per Match by Season')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f'{league_name.lower().replace(" ", "_")}_goals_trend.png')
    plt.close()
    
    # 2. Result distribution over seasons
    plt.figure(figsize=(12, 6))
    result_dist = df.groupby('Season')['FTR'].value_counts(normalize=True).unstack()
    result_dist.plot(kind='bar', stacked=True)
    plt.title(f'{league_name} - Match Result Distribution by Season')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f'{league_name.lower().replace(" ", "_")}_results_dist.png')
    plt.close()
    
    # 3. Home advantage trend
    plt.figure(figsize=(12, 6))
    home_adv = df.groupby('Season').apply(
        lambda x: (x['FTR'] == 'H').mean()
    )
    home_adv.plot(title=f'{league_name} - Home Win Percentage by Season')
    plt.axhline(y=home_adv.mean(), color='r', linestyle='--', label='Average')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f'{league_name.lower().replace(" ", "_")}_home_advantage.png')
    plt.close()
    
    # 4. Goals distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='FTHG', label='Home Goals', alpha=0.5)
    sns.histplot(data=df, x='FTAG', label='Away Goals', alpha=0.5)
    plt.title(f'{league_name} - Goal Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{league_name.lower().replace(" ", "_")}_goals_dist.png')
    plt.close()

def analyze_all_leagues():
    """Analyze data for all leagues"""
    leagues = {
        ('england', 'E0'): 'English Premier League',
        ('scotland', 'SC0'): 'Scottish Premier League',
        ('germany', 'D1'): 'German Bundesliga',
        ('italy', 'I1'): 'Italian Serie A',
        ('spain', 'SP1'): 'Spanish La Liga',
        ('france', 'F1'): 'French Ligue 1'
    }
    
    all_summaries = {}
    
    for (country, division), league_name in leagues.items():
        print(f"\nProcessing {league_name}...")
        
        # Try loading with different encodings if needed
        df = load_league_data(country, division)
        if df is None:
            print(f"Failed to load data for {league_name}")
            continue
        
        # Analyze trends
        summary = analyze_league_trends(df, league_name)
        all_summaries[league_name] = summary
        
        # Create visualizations
        plot_league_trends(df, league_name)
    
    return all_summaries

def prepare_training_data(df):
    """Prepare data for model training"""
    # Create feature matrix
    X = pd.DataFrame()
    
    # Basic match features
    X['HTHG'] = df['HTHG']
    X['HTAG'] = df['HTAG']
    
    # Add shot-based features if available
    if 'HS' in df.columns and 'AS' in df.columns:
        X['HS'] = df['HS']
        X['AS'] = df['AS']
    if 'HST' in df.columns and 'AST' in df.columns:
        X['HST'] = df['HST']
        X['AST'] = df['AST']
    
    # Add card-based features if available
    if 'HR' in df.columns and 'AR' in df.columns:
        X['HR'] = df['HR']
        X['AR'] = df['AR']
    
    # Create target variable
    y = df['FTR']
    
    return X, y

if __name__ == "__main__":
    # Run analysis for all leagues
    summaries = analyze_all_leagues() 