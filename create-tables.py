import pandas as pd
from random import choice


def ReduceToStandardSituations(pbp_data):
	"""Reduce play-by-play data to only plays that happen in standard game situations.

	Parameters
	----------
	pbp_data : DataFrame
		Play-by-play data fetched from nflverse. Required columns are — play_type, week, 
		half_seconds_remaining, score_differential, and penalty.

	Returns
	-------
	std_situations : DataFrame
		Reduced play-by-play data.
	"""

	valid_play_types = ['pass', 'run']
	std_situations = (
		pbp_data
		.query('play_type in @valid_play_types')
		.query('week < 15')
		# .query('half_seconds_remaining > 240')
		# .query('-15 < score_differential < 15')
		.query('penalty == 0') # TODO: handle penalties
	)
	return std_situations

def ReduceToFields(pbp_data, fields):
	"""Reduce play-by-play data to a user-selected set of fields.

	Parameters
	----------
	pbp_data : DataFrame 
		Play-by-play data fetched from nflverse. Required columns are — play_id, posteam, 
		defteam, week, and play_type.
	fields : list
		Columns the user wants to keep.
	
	Returns
	-------
	reduced : DataFrame 
		Reduced play-by-play data.
	"""

	req_fields = ['play_id', 'posteam', 'defteam', 'week', 'play_type']
	req_fields.extend(fields)
	reduced = pbp_data.loc[:, req_fields]
	return reduced

def CompileWeeklyTotals(pbp_data):
	"""Create a wide DataFrame with rows for team/week and cols for run/pass statistics.

	Parameters
	----------
	pbp_data : DataFrame 
		Play-by-play data fetched from nflverse. Required columns are — play_id, posteam, 
		defteam, week, and play_type.

	Returns
	-------
	weekly_totals : DataFrame 
		Counts and totals wrt team, offense/defense, week number, and play type 
		(in that order).
	"""

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
			.drop(columns=['play_id'])
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

def ComputeCSums(weekly_totals):
	"""Calculate the weekly cumulative sums for a set of weekly totals.

	Parameters
	----------
	weekly_totals : DataFrame 
		Weekly sums. Required to have a row multi-index where the first level is 
		team names, and the second level is week number.

	Returns
	-------
	weekly_csums : DataFrame 
		Weekly cumulative sums for each column.
	"""

	team_names = weekly_totals.index.get_level_values(0).unique().values
	weekly_csums = pd.concat(
		[weekly_totals.loc[name,:].cumsum() for name in team_names],
		keys=team_names,
		names=['team'],
	)
	return weekly_csums

def ConvertToCumulativeRates(weekly_csums):
	"""Convert weekly cumulative sums to weekly cumulative rates.

	Parameters
	----------
	weekly_csums : DataFrame 
		Weekly cumulative sums. Required to have a row multi-index where the first
		level is team names, and the second level is week num.

	Returns
	-------
	weekly_crates : DataFrame 
		Weekly cumulative rates for each column.
	"""

	col_names = weekly_csums.columns.values
	valid_play_types = [name.split('_',1)[0] for name in col_names]
	valid_play_types = list(set(valid_play_types))
	weekly_crates = pd.DataFrame(index=weekly_csums.index)
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
			weekly_crates = pd.concat([weekly_crates, rates], axis=1)
	return weekly_crates

def ComputeCumulativeRatesFromRawPBP(pbp_data):
	"""Convert a set of raw play-by-play data to a set of cumulative rates.
	
	Parameters
	----------
	pbp_data : DataFrame 
		Play-by-play data fetched from nflverse. Required columns are — play_id, 
		posteam, defteam, week, and play_type.

	Returns
	-------
	cumulative_rates : DataFrame 
		Weekly cumulative rates.
	"""

	weekly_totals = CompileWeeklyTotals(pbp_data)
	weekly_csums = ComputeCSums(weekly_totals)
	cumulative_rates = ConvertToCumulativeRates(weekly_csums)
	nonsense_cols = ['run_interception_pos', 'run_interception_def']
	cumulative_rates = cumulative_rates.drop(columns=nonsense_cols)
	return cumulative_rates

def ShuffleHomeAwayTeams(game_outcomes):
	"""Shuffle the home and away teams in a set of game outcomes.

	Game outcomes derived from nflverse play-by-play data always give home team
	columns first. We need to randomly swap the order and assign a new column to
	track which team is the home team.

	#TODO: needs serious optimization.

	Parameters
	----------
	game_outcomes : DataFrame
		Required columns are - game_id, home_team, away_team, total_home_score, 
		total_away_score, week.

	Returns
	-------
	shuffled_game_outcomes : DataFrame
		New frame with home and away teams shuffled.
	"""

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

def CreateLabeledTrainingData(shuffled_game_outcomes, cumulative_rates):
	"""Generate features by subtracting object team stats from adversary stats.
	
	Parameters
	----------
	shuffled_game_outcomes : DataFrame
		# TODO: required columns
	cumulative_rates : DataFrame
		# TODO: required columns

	Returns
	-------
	training_data : DataFrame
		A full set of labeled training data.
	"""

	def MakeFeature(cumulative_rates, row):
		"""Creates a set of features for a single nfl matchup

		Parameters
		----------
		cumulative_rates : DataFrame
		row : tuple
			A row of a DataFrame provided by itertuples.

		Returns
		-------
		feature : Series
			A row of engineered features for the nfl matchup
		"""

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

def FetchPlayByPlayData(season):
	"""Download play-by-play data from nflverse

	Parameters
	----------
	season : list
		The NFL season (1999-present) the user would like to query.

	Returns
	-------
	pbp_data : DataFrame
		Play-by-play data for every game of the queried season.
	"""
	url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
	pbp_data = pd.read_parquet(url)
	return pbp_data

if __name__ == "__main__":
	full_train_data = pd.DataFrame()
	for season in range(2011, 2022):
		print(f"fetching {season} season data")
		raw_data = FetchPlayByPlayData(season)
		standard_plays = ReduceToStandardSituations(raw_data)
		fields = ['epa', 'wpa', 'interception', 'fumble']
		standard_plays = ReduceToFields(standard_plays, fields)
		cumulative_rates = ComputeCumulativeRatesFromRawPBP(standard_plays)
		game_outcomes = (   # TODO: move this to its own function
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
