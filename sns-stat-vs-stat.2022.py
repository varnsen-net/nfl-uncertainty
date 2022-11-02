import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
from random import choices
import varnsen.tables as vtb
import varnsen.plot as vplt
import timeit



# preprocessing
sides = ['posteam', 'defteam']
metrics = ['epa', 'wpa']
data = pd.read_parquet('./season-data/2022-season.parquet')
win_rates = vtb.MapTeamWinRates(data)
data = vtb.ReduceToStandardSituations(data, half_seconds_remaining=0, sd=99)
data = data.loc[:,sides + metrics]
plot_data = vplt.CreatePlotData(data, sides, metrics)
print(plot_data)



# plot
fig, ax = plt.subplots(
	figsize = (4, 3.5),
)
x = plot_data.iloc[:,0] - plot_data.iloc[:,4]
y = plot_data.iloc[:,2] - plot_data.iloc[:,6]
# xerr = (plot_data.iloc[:,1]**2 + plot_data.iloc[:,5]**2) ** 0.5
# yerr = (plot_data.iloc[:,3]**2 + plot_data.iloc[:,7]**2) ** 0.5
# ax.errorbar(
	# x, y,
	# xerr=xerr.values,
	# yerr=yerr.values,
	# fmt='none',
	# alpha=0.2,
	# color='xkcd:dark grey',
	# zorder = 0.5,
# )
ax.axvline(x.mean(), color='xkcd:dark grey', zorder=0, linestyle='--', alpha=0.5)
ax.axhline(y.mean(), color='xkcd:dark grey', zorder=0, linestyle='--', alpha=0.5)
ax.scatter(x=x, y=y, alpha=0.0)
for name in win_rates.index:
	ax.text(
		x[name], y[name],
		s=name,
		ha='center',
		va='center',
		color=f"#{vplt.team_colors[name]}",
		fontweight='bold',
		fontsize=2 + (9 * win_rates[name]),
		zorder=1
	)
ax.set(
	# xticks = [],
	# yticks = [],
	# xlim = (-0.2, 0.2),
	# ylim = (0.3, -0.3),
	xlabel = 'EPA per play',
	ylabel = 'WPA per play',
	title = 'Average team WPA per play vs average team EPA per play\n(marker size is proportional to win/loss percentage)'
)
# ax.invert_yaxis()
# ax.axis('equal')


plt.tight_layout()
plt.savefig('./figures/stat-vs-stat.png')
# plt.show()
