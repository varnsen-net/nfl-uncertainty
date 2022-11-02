import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from preprocessing import FetchPlayByPlayData, ReduceToStandardSituations
import matplotlib as mpl
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



# sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
pal = sns.color_palette("hellafresh", 32)
g = sns.FacetGrid(
	plot_data,
	row=field,
	hue=field,
	aspect=15,
	height=.5,
	palette=pal,
	row_order=team_order.index,
	hue_order=team_order.index,
)

# Draw the densities in a few steps
g.map(sns.kdeplot, metric,
	bw_adjust=.73,
	clip_on=False,
	clip=(-5,5),
	fill=True,
	alpha=1,
	linewidth=1.5,
	# cut=0,
)
g.map(sns.kdeplot, metric,
	bw_adjust=.73,
	clip_on=False,
	clip=(-5,5),
	color=(0.99, 0.97, 0.87, 1),
	lw=2,
	# cut=0,
)
g.fig.suptitle("Distribution of Expected Points Added (EPA) per pass play")


# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
	ax = plt.gca()
	ax.text(0, .1, label, fontweight="bold", color=color,
			ha="left", va="center", transform=ax.transAxes)


g.map(label, metric)

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.55, left=0.05, right=0.95)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="", xlabel="EPA")
g.despine(bottom=True, left=True)

# plt.tight_layout()
plt.savefig('./figures/fig1.png')
