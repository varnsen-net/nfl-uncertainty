import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from varnsen.tables import RollingPythagExp, FetchLatestWeek
import requests
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import varnsen.plot as vplt
import varnsen.odds as vodds

# keys
odds_api_key = '6d64fa77e4e4a3fb13a2960d03824fd4'

# load pythagorean expectation data
pbp_data = pd.read_parquet('./pbp-data/2022-season.parquet')
pythag_exp = RollingPythagExp(pbp_data)
latest_pythag_exp = FetchLatestWeek(pythag_exp)

# transform pythagorean expectation data
scaler = load('./models/baseline-scaler.joblib')
latest_pythag_exp = (latest_pythag_exp - scaler.mean_) / scaler.scale_

# load odds data
response = vodds.FetchLeagueOddsFromAPI(odds_api_key, bookmaker='bovada')
current_odds = vodds.FetchCurrentOdds(response)
current_odds['is_home'] = 1
home_teams = current_odds['home'].values
away_teams = current_odds['away'].values
current_odds['obj_pyexp'] = latest_pythag_exp.loc[home_teams].values
current_odds['adv_pyexp'] = latest_pythag_exp.loc[away_teams].values

# predict upcoming game probabilities
model = load('./models/baseline-logistic-regression.joblib')
x = current_odds[['obj_pyexp', 'adv_pyexp', 'is_home']]
current_odds['home_prob'] = model.predict_proba(x)[:,1]
current_odds['away_prob'] = 1 - current_odds['home_prob']

# calculate implied probabilities and expected value for each bet
current_odds['home_implied'] = [vodds.CalculateImpliedOdds(odds) for odds in current_odds['home_line']]
current_odds['away_implied'] = [vodds.CalculateImpliedOdds(odds) for odds in current_odds['away_line']]
current_odds['home_edge'] = current_odds['home_prob'] - current_odds['home_implied']
current_odds['away_edge'] = current_odds['away_prob'] - current_odds['away_implied']

# plot predicted odds vs implied odds
home_data = current_odds[['home', 'home_prob', 'home_implied']]
home_data.columns = ['team', 'prob', 'implied']
away_data = current_odds[['away', 'away_prob', 'away_implied']]
away_data.columns = ['team', 'prob', 'implied']
scatter_data = pd.concat([home_data, away_data]).set_index('team')
print(scatter_data.index)
sns.set_palette('hellafresh')
fig, ax = plt.subplot_mosaic(
    "AB",
    figsize=(5.8,3),
)
ax['A'].plot([0,1], [0,1], '--', zorder=0)
for i in range(len(scatter_data)):
    name = scatter_data.index[i]
    ax['A'].text(
        scatter_data.iloc[i]['implied'], scatter_data.iloc[i]['prob'],
        s=name,
        ha='center',
        va='center',
        color=f"#{vplt.team_colors[name]}",
        fontweight='bold',
        fontsize=5,
        zorder=1,
    )
ax['A'].set(
    xlim=(0,1),
    ylim=(0,1),
    xlabel = 'Implied probability',
    ylabel = 'Predicted probability',
    title = 'Model-generated probabilities vs implied-odds probabilities\nfor NFL week 10 matchups',
)

# bar plot of home and away edge
bar_data = scatter_data['prob'] - scatter_data['implied']
bar_data = bar_data.sort_values()
print(bar_data)
drange = range(1,len(bar_data)+1)
ax['B'].barh(drange, bar_data, tick_label=bar_data.index, linewidth=0)
ax['B'].set(
    xlabel = 'Expected value per betting unit',
    ylabel = 'Team',
    title = 'Expected value of available week 10 moneylines',
)

plt.tight_layout()
plt.savefig('./figures/week-10-probabilities.png')
# plt.show()
