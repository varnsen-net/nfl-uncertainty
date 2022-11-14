import pandas as pd
import requests

# import team name mapping
teams = pd.read_csv('./errata/nfl-teams.csv', index_col=0)
team_name_map = dict(zip(teams['Name'], teams['Abbreviation']))

# functions
def FetchLeagueOddsFromAPI(odds_api_key, bookmaker='bovada'):
    """
    Fetches NFL moneyline odds from the Odds API for the given bookmaker.
    
    Parameters
    ----------
    odds_api_key : str
        API key for the Odds API — https://the-odds-api.com/
    bookmaker : str
        Bookmaker to fetch odds from. Default is 'bovada'.
        
    Returns
    -------
    response : Response
        Response object from the Odds API.
    """
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    markets = 'h2h'
    odds_format = 'american'
    date_format = 'iso'
    response = requests.get(
        url = url,
        params = {
            'api_key' : odds_api_key,
            'markets' : markets,
            'oddsFormat' : odds_format,
            'dateFormat' : date_format,
            'bookmakers' : bookmaker,
        }
    )
    return response

def ExtractGameInfo(game_odds):
    """
    Extracts team names and odds for an NFL matchup.
    
    Parameters
    ----------
    game_odds : dict
        Dictionary containing info for an NFL matchup.
    
    Returns
    -------
    game_info : tuple
        Tuple containing team names and odds for an NFL matchup.
    """
    tipoff = game_odds['commence_time']
    home_team = game_odds['home_team']
    away_team = game_odds['away_team']
    markets = game_odds['bookmakers'][0]['markets'][0]
    h2h_outcomes = markets['outcomes']
    home_h2h = [f['price'] for f in h2h_outcomes if f['name'] == home_team][0]
    away_h2h = [f['price'] for f in h2h_outcomes if f['name'] == away_team][0]
    game_info = (tipoff, team_name_map[home_team], home_h2h, team_name_map[away_team], away_h2h)
    return game_info

def FetchCurrentOdds(response):
    """
    Extracts game info for every NFL matchup found in an Odds API response,
    then packages it all into a DataFrame.
    
    Parameters
    ----------
    response : Response
        Response object from the Odds API.
    
    Returns
    -------
    current_odds : pd.DataFrame
        DataFrame containing game info for every NFL matchup found in an Odds
        API response.
    """
    games = response.json()
    games = [g for g in games if len(g['bookmakers']) > 0]
    current_odds = [ExtractGameInfo(g) for g in games]
    current_odds = pd.DataFrame(current_odds)
    current_odds.columns = ['tipoff', 'home', 'home_line', 'away', 'away_line']
    return current_odds

def CalculateImpliedOdds(odds:int) -> float:
    """Calculate the implied probability of an outcome given American odds."""
    if odds < 0:
        odds = abs(odds)
        implied_odds = odds / (100 + odds)
    else:
        implied_odds = 100 / (100 + odds)
    return implied_odds

