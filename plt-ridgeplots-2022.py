import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import FetchPlayByPlayData, ReduceToStandardSituations
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import varnsen_theme

data = FetchPlayByPlayData(2022)
data = ReduceToStandardSituations(data)
field = 'posteam'
metric = 'epa'

# preprocessing
plot_data = (
	data
	.query('play_type in ["pass"]')
	.loc[:,[field, metric]]
)
team_order = plot_data.groupby(field).mean().sort_values(metric, ascending=False)
# print(team_order)
print(plot_data)

# fit kde with grid search for optimal bandwidth
# shufsplit = ShuffleSplit(n_splits=10, test_size=0.25)
# grid = GridSearchCV(
	# KernelDensity(),
	# {'bandwidth' : np.linspace(0.1, 1.0, 50)},
	# cv = shufsplit,
	# refit=True,
# )

x_plot = np.linspace(-5,5,2000)[:, np.newaxis]
kde = KernelDensity(bandwidth=0.275)
kde.fit(plot_data['epa'].values.reshape(-1,1))
log_dens_nfl = kde.score_samples(x_plot)

# plot
fig, axs = plt.subplots(
	figsize = (5,8),
	dpi = 120,
	nrows = 32,
	ncols = 1,
	sharex = True,
)
team_axs = zip(team_order.index, axs)
for t in team_axs:
	team = t[0]
	ax = t[1]
	team_data = plot_data.query('posteam == @team')
	kde = KernelDensity(bandwidth=0.275)
	kde.fit(team_data['epa'].values.reshape(-1,1))
	log_dens_team = kde.score_samples(x_plot)
	y_data = np.exp(log_dens_team)
	# y_data = np.exp(log_dens_team) - np.exp(log_dens_nfl)
	ax.plot(x_plot, y_data)
	ax.set(yticks=[], ylabel="")
	ax.spines[:].set_visible(False)
plt.subplots_adjust(hspace=-.55, left=0.05, right=0.95)
# ax.plot(plot_data['epa'], -0.005 - 0.01 * np.random.random(plot_data['epa'].shape[0]), "+k")
# plt.tight_layout()
plt.show()

