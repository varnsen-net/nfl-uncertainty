import pandas as pd
from random import choice


def ReduceToStandardSituations(pbp_data: pd.DataFrame) -> pd.DataFrame:
	"""Reduce play-by-play data to only plays that happen in standard game situations"""
	valid_play_types = ['pass', 'run']
	std_situations = (
		pbp_data
		# .query('week < 15')
		# .query('half_seconds_remaining > 240')
		.query('play_type in @valid_play_types')
		# .query('-15 < score_differential < 15')
		.query('penalty == 0') # TODO: handle penalties
	)
	return std_situations

def CompileWeeklyTotals(pbp_data: pd.DataFrame, fields: list) -> pd.DataFrame:
	"""Create a wide DataFrame with rows for team/week and cols for run/pass statistics"""
	weekly_totals = pd.DataFrame()
	for side in ['posteam', 'defteam']:
		play_grouping = pbp_data.groupby([side, 'week', 'play_type'])
		counts = (
			play_grouping
			[[side, 'week', 'play_type', 'play_id']] # only need to count values for one col
			.count()
			.iloc[:,0]
			.rename('count')
		)
		sums = (
			play_grouping
			.sum(numeric_only=True)
			.loc[:, fields]
		)
		totals = pd.concat([counts, sums], axis=1)
		# flatten the column multi-index
		totals = totals.unstack(level='play_type')
		totals.columns = [
			f"{c[1].replace('_', '')}_{c[0]}_{side[:3]}" 
			for c in totals.columns.values
		]
		weekly_totals = pd.concat([weekly_totals, totals], axis=1)
	return weekly_totals

def ComputeCSums(weekly_totals: pd.DataFrame) -> pd.DataFrame:
	"""Reduce """
	team_names = weekly_totals.index.get_level_values(0).unique().values
	team_csums = pd.concat(
		[weekly_totals.loc[name,:].cumsum() for name in team_names],
		keys=team_names,
		names=['team'],
	)
	return team_csums

def ConvertToCumulativeRates(weekly_csums: pd.DataFrame) -> pd.DataFrame:
	"""Reduce """
	col_names = weekly_csums.columns.values
	valid_play_types = [name.split('_',1)[0] for name in col_names]
	valid_play_types = list(set(valid_play_types))
	all_rates = pd.DataFrame(index=weekly_csums.index)
	for side in ['pos', 'def']:
		side_csums = weekly_csums.filter(like=f"_{side}")
		for p in valid_play_types:
			csums = side_csums.filter(like=f"{p}_")
			counts = (
				csums
				.filter(like=f"{p}_count")
				.values
			)
			rates = (
				csums
				.divide(counts, axis=1)
				.iloc[:,1:]
			)
			all_rates = pd.concat([all_rates, rates], axis=1)
	return all_rates

def ComputeCumulativeRatesFromRawPBP(pbp_data: pd.DataFrame, fields: list) -> pd.DataFrame:
	"""Main function"""
	weekly_totals = CompileWeeklyTotals(standard_plays, fields)
	weekly_csums = ComputeCSums(weekly_totals)
	cumulative_rates = ConvertToCumulativeRates(weekly_csums)
	nonsense_cols = ['run_interception_pos', 'run_interception_def']
	cumulative_rates = cumulative_rates.drop(columns=nonsense_cols)
	return cumulative_rates

def ShuffleHomeAwayTeams(game_outcomes: pd.DataFrame) -> pd.DataFrame:
	"""Probably really slow on a big dataset"""
	shuffled_game_outcomes = [
		(row[0], row[2], row[1], row[4], row[3], row[5], 0)
		if choice([True, False])
		else row + (1,)
		for row in game_outcomes.itertuples()
	]
	colnames = ['game_id', 'obj_team', 'adversary', 'obj_score', 'adversary_score', 'week', 'is_home']
	shuffled_game_outcomes = (
		pd.DataFrame(shuffled_game_outcomes, columns=colnames)
		.set_index('game_id')
	)
	return shuffled_game_outcomes

def CreateLabeledTrainingData(shuffled_game_outcomes: pd.DataFrame, cumulative_rates: pd.DataFrame) -> pd.DataFrame:
	"""
	Subtract objective team stats from adversary stats to generate features
	"""
	def MakeFeature(cumulative_rates: pd.DataFrame, row: pd.Series) -> pd.Series:
		"""Creates a set of features for a given nfl game"""
		obj_team = row[1]
		adversary = row[2]
		week_num = row[5]
		obj_prior_week_idx = cumulative_rates.loc[obj_team].index.get_loc(week_num) - 1
		adv_prior_week_idx = cumulative_rates.loc[adversary].index.get_loc(week_num) - 1
		obj_team_feat = cumulative_rates.loc[obj_team].iloc[obj_prior_week_idx]
		adv_team_feat = cumulative_rates.loc[adversary].iloc[adv_prior_week_idx]
		feature = obj_team_feat - adv_team_feat
		return feature

	features = [
		MakeFeature(cumulative_rates, row)
		for row in shuffled_game_outcomes.itertuples()
	]
	features = pd.DataFrame(features, index=shuffled_game_outcomes.index)
	training_data = pd.concat([shuffled_game_outcomes, features], axis=1)
	score_diffs = training_data['obj_score'] - training_data['adversary_score']
	training_data['obj_team_win'] = [1 if s > 0 else 0 for s in score_diffs]
	training_data = training_data.iloc[:,5:]
	return training_data

full_train_data = pd.DataFrame()
for year in range(2010, 2022):
	print(f"fetching {year} data")
	raw_data = pd.read_parquet(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet")
	standard_plays = ReduceToStandardSituations(raw_data)
	cumulative_rates = ComputeCumulativeRatesFromRawPBP(standard_plays, ['epa', 'wpa', 'interception', 'fumble'])
	game_outcomes = (
		raw_data
		.groupby('game_id').last()
		[['home_team', 'away_team', 'total_home_score', 'total_away_score', 'week']]
		.query('4 <= week <=14')
	)
	shuffled_game_outcomes = ShuffleHomeAwayTeams(game_outcomes)
	training_data = CreateLabeledTrainingData(shuffled_game_outcomes, cumulative_rates)
	full_train_data = pd.concat([full_train_data, training_data])
print(full_train_data)
# full_train_data.to_csv('./train.csv')
