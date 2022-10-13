import pandas as pd

data = pd.read_parquet('./data/play_by_play_2020.parquet')
# data.query('game_id == "2020_01_ARI_SF"').to_csv('./2020_01_ARI_SF.csv')
# [print(f"'{c}',") for c in data.columns]
cols = [
	'play_id',
	'game_id',
	'home_team',
	'away_team',
	'week',
	'posteam',
	'defteam',
	'half_seconds_remaining',
	'score_differential',
	'play_type',
	'penalty',
	'fumble',
	'interception',
	'epa',
	'wpa',
]
# data = data[cols]
# data['drive_id'] = data['game_id'] + '_' + data['drive'].astype(str)


def ReduceToStandardSituations(pbp_data: pd.DataFrame) -> pd.DataFrame:
	"""Reduce play-by-play data to only plays that happen in standard game situations"""
	valid_play_types = ['pass', 'run']
	std_situations = (
		pbp_data
		.query('week < 15')
		.query('half_seconds_remaining > 240')
		.query('play_type in @valid_play_types')
		.query('-15 < score_differential < 15')
		.query('penalty == 0') # TODO: handle penalties
	)
	return std_situations

def SeparateOffAndDefPlays(pbp_data: pd.DataFrame) -> list:
	"""Create two pandas groupby objects — one for offensive plays, and one for defensive plays"""
	off_group = pbp_data.groupby(['posteam', 'week', 'play_type'])
	def_group = pbp_data.groupby(['defteam', 'week', 'play_type'])
	return off_group, def_group

def CompileWeeklyTotals(play_grouping: pd.DataFrame.groupby, prefix: str) -> pd.DataFrame:
	"""Create a wide DataFrame with rows for team/week and cols for run/pass statistics"""
	counts = (
		play_grouping
		[['posteam', 'week', 'play_type', 'epa']] # only need to count values for one col
		.count()
		.iloc[:,0]
		.rename('count')
	)
	sums = (
		play_grouping
		.sum(numeric_only=True)
		.loc[:, ['epa', 'wpa', 'interception', 'fumble']]
	)
	outcomes = pd.concat([counts, sums], axis=1)
	# flatten the column multi-index
	outcomes = outcomes.unstack(level='play_type')
	outcomes.columns = [
		f"{c[1].replace('_', '')}_{c[0]}_{prefix}" 
		for c in outcomes.columns.values
	]
	return outcomes

def ComputeCSums(weekly_totals: pd.DataFrame) -> pd.DataFrame:
	"""Reduce """
	team_names = weekly_totals.index.get_level_values(0).unique().values
	team_csums = pd.concat(
		[weekly_totals.loc[name,:].cumsum() for name in team_names],
		keys=team_names,
		names=['team'],
	)
	return team_csums

def ConvertToRates(weekly_csums: pd.DataFrame) -> pd.DataFrame:
	"""Reduce """
	col_names = weekly_csums.columns.values
	valid_play_types = [name.split('_',1)[0] for name in col_names]
	valid_play_types = list(set(valid_play_types))
	all_rates = pd.DataFrame(index=weekly_csums.index)
	for p in valid_play_types:
		csums = weekly_csums.filter(like=f"{p}_")
		counts = (
			weekly_csums
			.filter(like=f"{p}_count")
			.values
		)
		rates = (csums
			.divide(counts, axis=1)
			.iloc[:,1:]
		)
		all_rates = pd.concat([all_rates, rates], axis=1)
	return all_rates


def ComputeWeeklyRatesFromSeasonPBP(pbp_data: pd.DataFrame) -> pd.DataFrame:
	"""Main function"""
	standard_plays = ReduceToStandardSituations(data)
	off_group, def_group = SeparateOffAndDefPlays(standard_plays)
	weekly_totals_off = CompileWeeklyTotals(off_group, 'o')
	weekly_totals_def = CompileWeeklyTotals(def_group, 'd')
	weekly_csums_off = ComputeCSums(weekly_totals_off)
	weekly_csums_def = ComputeCSums(weekly_totals_def)
	rates_by_week_off = ConvertToRates(weekly_csums_off)
	rates_by_week_def = ConvertToRates(weekly_csums_def)
	rates_by_week_all = (
		pd.concat([rates_by_week_off, rates_by_week_def], axis=1)
		.drop(columns=['run_interception_o', 'run_interception_d'])
	)
	return rates_by_week_all

rates_by_week_all = ComputeWeeklyRatesFromSeasonPBP(data)

games = data.groupby('game_id').last()[['home_team', 'away_team', 'week', 'score_differential']]
print(games)
