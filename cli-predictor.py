import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from varnsen.tables import RollingPythagExp
import requests
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import varnsen.plot as vplt

# keys
odds_api_key = 'b8b2a510ac6e624db9398f91de84a338'

# load data
pbp_data = pd.read_parquet('./pbp-data/2022-season.parquet')
teams = pd.read_csv('./errata/nfl-teams.csv')
team_name_map = dict(zip(teams['Name'], teams['Abbreviation']))

# TODO: move these functions to a separate file
def FetchLeagueOddsFromAPI(odds_api_key):
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    regions = 'us'
    markets = 'h2h'
    odds_format = 'american'
    date_format = 'iso'
    bookmakers = 'bovada'
    response = requests.get(
        url = url,
        params = {
            'api_key' : odds_api_key,
            # 'regions' : regions,
            'markets' : markets,
            'oddsFormat' : odds_format,
            'dateFormat' : date_format,
            'bookmakers' : bookmakers,
        }
    )
    return response

def FetchLatestWeek(data:pd.DataFrame) -> pd.DataFrame:
    team_names = data.index.get_level_values(0).unique()
    most_recent_data = pd.concat([data.loc[name].tail(1) for name in team_names])
    most_recent_data.index = team_names
    return most_recent_data

def ExtractGameInfo(game_odds:dict):
    tipoff = game_odds['commence_time']
    home_team = game_odds['home_team']
    away_team = game_odds['away_team']
    markets = game_odds['bookmakers'][0]['markets'][0]
    h2h_outcomes = markets['outcomes']
    home_h2h = [f['price'] for f in h2h_outcomes if f['name'] == home_team][0]
    away_h2h = [f['price'] for f in h2h_outcomes if f['name'] == away_team][0]
    return (tipoff, team_name_map[home_team], home_h2h, team_name_map[away_team], away_h2h)

def FetchCurrentOdds(response):
    games = response.json()
    games = [g for g in games if len(g['bookmakers']) > 0]
    current_odds = [ExtractGameInfo(g) for g in games]
    current_odds = pd.DataFrame(current_odds)
    current_odds.columns = ['tipoff', 'home', 'home_line', 'away', 'away_line']
    return current_odds

def CalculateImpliedOdds(odds:int):
    if odds < 0:
        odds = abs(odds)
        implied_odds = odds / (100 + odds)
    else:
        implied_odds = 100 / (100 + odds)
    return implied_odds

if __name__ == "__main__":
    scaler = load('./models/baseline-scaler.joblib')
    pythag_exp = RollingPythagExp(pbp_data)
    latest_pythag_exp = FetchLatestWeek(pythag_exp)
    latest_pythag_exp = (latest_pythag_exp - scaler.mean_) / scaler.scale_
    response = FetchLeagueOddsFromAPI(odds_api_key)
    current_odds = FetchCurrentOdds(response)
    current_odds['is_home'] = 1
    home_teams = current_odds['home'].values
    away_teams = current_odds['away'].values
    current_odds['obj_pyexp'] = latest_pythag_exp.loc[home_teams].values
    current_odds['adv_pyexp'] = latest_pythag_exp.loc[away_teams].values
    model = load('./models/baseline-logistic-regression.joblib')
    x = current_odds[['obj_pyexp', 'adv_pyexp', 'is_home']]
    current_odds['home_prob'] = model.predict_proba(x)[:,1]
    current_odds['away_prob'] = 1 - current_odds['home_prob']
    current_odds['home_implied'] = [CalculateImpliedOdds(odds) for odds in current_odds['home_line']]
    current_odds['away_implied'] = [CalculateImpliedOdds(odds) for odds in current_odds['away_line']]
    current_odds['home_edge'] = current_odds['home_prob'] - current_odds['home_implied']
    current_odds['away_edge'] = current_odds['away_prob'] - current_odds['away_implied']
    print(current_odds)
    
    sns.set_palette('hellafresh')
    fig, ax = plt.subplots(figsize=(3.5,3))
    ax.plot([0,1], [0,1])
    ax.plot(current_odds['home_implied'], current_odds['home_edge']+current_odds['home_implied'], 'o', ms=2, label='Home')
    ax.plot(current_odds['away_implied'], current_odds['away_edge']+current_odds['away_implied'], 'o', ms=2, label='Away')
    plt.tight_layout()
    plt.show()
