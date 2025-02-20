import requests
import pandas as pd
from pathlib import Path
import os

def generate_season_codes(start_year=10, end_year=23):
    """Generate season codes from 2010/11 to current"""
    seasons = []
    for year in range(start_year, end_year + 1):
        season = f"{year:02d}{year+1:02d}"
        seasons.append(season)
    return seasons

def fetch_football_data(seasons=None):
    """Fetch data for major leagues across multiple seasons"""
    # Base URL and country codes
    base_url = "https://www.football-data.co.uk/"
    countries = {
        'england': 'englandm.php',
        'scotland': 'scotlandm.php',
        'germany': 'germanym.php',
        'italy': 'italym.php',
        'spain': 'spainm.php',
        'france': 'francem.php'
    }
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # If no seasons specified, get all from 2010/11 to current
    if seasons is None:
        seasons = generate_season_codes()
    
    # Add current season
    if "2324" not in seasons:
        seasons.append("2324")
    
    # Main division file codes
    divisions = {
        'england': ['E0'],  # Premier League
        'scotland': ['SC0'],  # Scottish Premier League
        'germany': ['D1'],   # Bundesliga
        'italy': ['I1'],     # Serie A
        'spain': ['SP1'],    # La Liga
        'france': ['F1']     # Ligue 1
    }
    
    for country, page in countries.items():
        print(f"\nFetching {country} data...")
        country_dir = data_dir / f"{country}-data"
        country_dir.mkdir(exist_ok=True)
        
        for season in seasons:
            print(f"\nProcessing season {season}...")
            
            for div in divisions[country]:
                csv_url = f"{base_url}mmz4281/{season}/{div}.csv"
                output_file = country_dir / f"{div}_{season}.csv"
                
                # Skip if file already exists
                if output_file.exists():
                    print(f"File already exists: {output_file}")
                    continue
                
                try:
                    # Fetch CSV data
                    print(f"Downloading {csv_url}...")
                    response = requests.get(csv_url)
                    response.raise_for_status()
                    
                    # Save to file
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded to {output_file}")
                    
                    # Read and preview the data
                    try:
                        df = pd.read_csv(output_file)
                        print(f"Total matches: {len(df)}")
                    except pd.errors.EmptyDataError:
                        print("Warning: Empty or invalid CSV file")
                        if output_file.exists():
                            output_file.unlink()  # Delete empty/invalid file
                    except Exception as e:
                        print(f"Error reading CSV: {e}")
                        if output_file.exists():
                            output_file.unlink()  # Delete problematic file
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {div} data for season {season}: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")

def main():
    # Fetch all seasons from 2010/11 to current
    fetch_football_data()

if __name__ == "__main__":
    main() 