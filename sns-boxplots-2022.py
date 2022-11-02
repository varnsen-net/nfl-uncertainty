import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import FetchPlayByPlayData, ReduceToStandardSituations
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
print(team_order)
print(plot_data)

# plot
sns.set_theme('notebook')
pal = sns.color_palette("hellafresh", 32)
fig, ax = plt.subplots(
	figsize = (6,8),
	dpi = 120,
)
sns.boxplot(
	plot_data,
	x = metric,
	y = field,
	order = team_order.index,
	palette = pal,
	showfliers = False,
	ax = ax,
)
# sns.kdeplot(
	# plot_data,
	# x = metric,
	# hue = field,
	# alpha = 0.5,
	# hue_order = team_order.index,
	# palette = pal,
	# legend = False,
	# ax = ax,
# )
ax.set_title('The Kansas City Football Chiefs of Football\n still have the best passing attack')
ax.set_xlabel('Expected Points Added (EPA)')
ax.set_ylabel('Team')

plt.tight_layout()
plt.show()
