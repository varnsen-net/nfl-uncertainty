import pandas as pd
from numpy import mean as npmean
from scipy.stats import bootstrap
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def hex_to_rgb(hex_codes: list) -> list:
    """Convert a list of hex color codes to a list of normed rgb tuples"""
    l = lambda x: [int(x[i:i+2], 16)/256 for i in (0,2,4)]
    rgb_list = [tuple(l(code)) for code in hex_codes]
    return rgb_list

def CreatePlotData(data:pd.DataFrame, sides:list, metrics:list) -> pd.DataFrame:
    """Create a DataFrame of bootstrapped stats for each team"""
    col_pairs = [(s,m) for s in sides for m in metrics]
    team_names = data[sides[0]].unique()
    plot_data = pd.DataFrame(index=team_names).sort_index()
    for s,m in col_pairs:
        stats = []
        for team in team_names:
            team_data = data[data[s] == team][m]
            mean = team_data.mean()
            boot = bootstrap(
                (team_data, ),
                npmean,
                n_resamples = 1000,
                method = 'percentile',
            )
            row = (team, mean, boot.standard_error)
            stats.append(row)
        stats = (
            pd.DataFrame(stats)
            .set_index(0)
            .sort_index()
        )
        stats.columns = [f"{s}_{m}", f"{s}_{m}_err"]
        plot_data = pd.concat([plot_data, stats], axis=1)
    return plot_data 

team_colors = {
    'ARI' : '97233F',
    'ATL' : 'A71930',
    'BAL' : '00338D',
    'BUF' : '00338D',
    'CAR' : '0085CA',
    'CHI' : '0B162A',
    'CIN' : 'FB4F14',
    'CLE' : 'FB4F14',
    'DAL' : '002244',
    'DEN' : 'FB4F14',
    'DET' : '005A8B',
    'GB' : '203731',
    'HOU' : 'A71930',
    'IND' : '002C5F',
    'JAX' : '006778',
    'KC' : 'E31837',
    'LAC' : '0073CF',
    'LA' : '002244',
    'MIA' : '008E97',
    'MIN' : '4F2683',
    'NE' : '002244',
    'NO' : '000000',
    'NYG' : '0B2265',
    'NYJ' : '203731',
    'LV' : '000000',
    'PHI' : '004953',
    'PIT' : '000000',
    'SF' : 'AA0000',
    'SEA' : '002244',
    'TB' : 'D50A0A',
    'TEN' : '002244',
    'WAS' : '773141',
}

palettes = {
    "vaporwave": ["94D0FF", "8795E8", "966bff", "AD8CFF", "C774E8", "c774a9", "FF6AD5", "ff6a8b", "ff8b8b", "ffa58b", "ffde8b", "cdde8b", "8bde8b", "20de8b"],
    "vcool": ["FF6AD5", "C774E8", "AD8CFF", "8795E8", "94D0FF"],
    "crystal_pepsi": ["FFCCFF", "F1DAFF", "E3E8FF", "CCFFFF"],
    "mallsoft": ["fbcff3", "f7c0bb", "acd0f4", "8690ff", "30bfdd", "7fd4c1"],
    "jazzcup": ["392682", "7a3a9a", "3f86bc", "28ada8", "83dde0"],
    "sunset": ["661246", "ae1357", "f9247e", "d7509f", "f9897b"],
    "macplus": ["1b4247", "09979b", "75d8d5", "ffc0cb", "fe7f9d", "65323e"],
    "seapunk": ["532e57", "a997ab", "7ec488", "569874", "296656"],
    "avanti": ["FB4142", "94376C", "CE75AD", "76BDCF", "9DCFF0"],
    "hellafresh" : ['2f4858', '33658a', '86bbd8', 'f6ae2d', 'f26419'],
}
palettes = {k : hex_to_rgb(palettes[k]) for k in palettes.keys()}
for k in palettes.keys():
    cmap = LinearSegmentedColormap.from_list(k, palettes[k])
    mpl.colormaps.register(cmap)


rcparams = {
    "axes.facecolor" : (0, 0, 0, 0),
    "figure.facecolor" : "efe8db",
    # "figure.facecolor" : "f3edde",
    # "figure.facecolor" : "ede0d4",
    # "figure.facecolor" : "FEFAE0",
    # "figure.figsize" : (1,1),
    "figure.dpi" : 300,
    "lines.linewidth" : 0.6,
    'xtick.bottom' : True,
    'ytick.left' : True,
}
sns.set_theme(
    context = 'paper',
    style="white",
    font_scale = 0.60,
    rc=rcparams,
)
