import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import varnsen.tables as vtb
import varnsen.plot as vplt



# preprocessing
final_data = pd.DataFrame()
sides = ['posteam', 'defteam']
metrics = ['wpa']
for year in range(2016, 2022):
	print(f"fetching {year} season data")
	season_data = pd.DataFrame()
	data = vtb.FetchPlayByPlayData(year)
	win_rates = vtb.MapTeamWinRates(data)
	data = vtb.ReduceToStandardSituations(data)
	data = data.loc[:,sides + metrics]
	epa_data = vplt.CreatePlotData(data, sides, metrics)
	season_data['x'] = epa_data.iloc[:,0] - epa_data.iloc[:,2]
	season_data['y'] = win_rates
	final_data = pd.concat([final_data, season_data])
print(final_data)


# plot
fig, ax = plt.subplots(
	figsize = (4, 3.5),
)
# x = plot_data.iloc[:,0] - plot_data.iloc[:,4]
# y = plot_data.iloc[:,2] - plot_data.iloc[:,6]
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
# ax.axvline(x.mean(), color='xkcd:dark grey', zorder=0, linestyle='--', alpha=0.5)
# ax.axhline(y.mean(), color='xkcd:dark grey', zorder=0, linestyle='--', alpha=0.5)
ax.scatter(x=final_data['x'], y=final_data['y'])
# ax.set(
	# # xticks = [],
	# # yticks = [],
	# # xlim = (-0.2, 0.2),
	# # ylim = (0.3, -0.3),
	# xlabel = 'EPA per play',
	# ylabel = 'WPA per play',
	# title = 'Average team WPA per play vs average team EPA per play\n(marker size is proportional to win/loss percentage)'
# )
# ax.invert_yaxis()
# ax.axis('equal')


plt.tight_layout()
plt.savefig('./figures/winloss-vs-epa.png')
# plt.show()
