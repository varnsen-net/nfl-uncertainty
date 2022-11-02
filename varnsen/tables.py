import pandas as pd
from random import choice

def FetchPlayByPlayData(season):
	"""
	Download play-by-play data from nflverse

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

def ReduceToStandardSituations(
	pbp_data, 
	week_nums = (0, 99),
	half_seconds_remaining = 120,
	sd = 14,
):
	"""
	Reduce play-by-play data to only plays that happen in standard game situations.

	Parameters
	----------
	pbp_data : DataFrame
		Play-by-play data fetched from nflverse. Required columns are — play_type, week, 
		half_seconds_remaining, score_differential, and penalty.
	week_num : int
		NFL week number.
	half_seconds_remaining : int
		Number of seconds remaining in the half at the start of a play.
	sd : int
		Score differential at the start of a play. The delta is always with respect 
		to posteam, i.e. posteam score - defteam score.

	Returns
	-------
	std_situations : DataFrame
		Reduced play-by-play data.
	"""
	w_low, w_high = week_nums
	non_play_types = ['TIMEOUT', 'GAME_START', 'END_QUARTER', 'END_GAME']
	std_situations = (
		pbp_data
		.query('special_teams_play == 0')
		.query('play_type_nfl not in @non_play_types')
		.query(f"{w_low} <= week <= {w_high}")
		.query('half_seconds_remaining > @half_seconds_remaining')
		.query(f"-{sd} <= score_differential <= {sd}")
		# .query('penalty == 0') # TODO: handle penalties
	)
	return std_situations

def ReduceToFields(pbp_data, fields):
	"""
	Reduce play-by-play data to a user-selected set of fields.

	Parameters
	----------
	pbp_data : DataFrame 
		Play-by-play data fetched from nflverse. Required columns are — play_id,
		posteam, defteam, week, and play_type.
	fields : list
		Columns the user wants to keep.
	
	Returns
	-------
	reduced : DataFrame 
		Reduced play-by-play data.
	"""

	req_fields = ['posteam', 'defteam', 'week']
	req_fields.extend(fields)
	reduced = pbp_data[req_fields]
	return reduced

def ShuffleHomeAwayTeams(game_outcomes):
	"""
	Shuffle the home and away teams in a set of game outcomes.

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
	"""
	Generate features by subtracting object team stats from adversary stats.
	
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
		"""
		Creates a set of features for a single nfl matchup

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

def MapTeamWinRates(data:pd.DataFrame) -> dict:
	"""Calculate the win-loss percentage for each team"""
	cols = ['game_id', 'home_team', 'away_team', 'total_home_score', 'total_away_score']
	data = (
		data
		.loc[:,cols]
		.dropna()
	)
	outcomes = data.groupby('game_id').last()
	outcomes['score_delta'] = outcomes['total_home_score'] - outcomes['total_away_score']
	outcomes = outcomes.query('score_delta != 0')
	h_wins = outcomes.query('score_delta > 0').home_team.value_counts().sort_index()
	a_wins = outcomes.query('score_delta < 0').away_team.value_counts().sort_index()
	wins = h_wins.add(a_wins, fill_value=0)
	totals = pd.concat([outcomes['home_team'], outcomes['away_team']]).value_counts()
	win_rates = wins.div(totals, fill_value=0)
	return win_rates

def RollingMeans(data:pd.DataFrame) -> pd.DataFrame:
	"""TODO"""
	posteam = data.drop(columns='defteam')
	defteam = data.drop(columns='posteam')
	season_rolling_means = pd.DataFrame()
	for df in [posteam, defteam]:
		side = df.columns[0]
		grouped = (
			df	
			.groupby([side, 'week'])
			.agg(func=['count', 'sum'])
		)
		rolling_stats = pd.DataFrame()
		for name in grouped.index.get_level_values(0).unique():
			rolled = grouped.loc[name].expanding().sum()
			new_idx = [(name,week) for week in rolled.index]
			rolled.index = pd.MultiIndex.from_tuples(new_idx)
			rolling_stats = pd.concat([rolling_stats, rolled])
		metrics = rolling_stats.columns.get_level_values(0).unique()
		rolling_means = pd.DataFrame()
		for m in metrics:
			rolling_means[f"{side}_{m}"] = rolling_stats[m]['sum'] / rolling_stats[m]['count']
		rolling_means.index.names = ['team', 'week']
		season_rolling_means = pd.concat([season_rolling_means, rolling_means], axis=1)
	return season_rolling_means

if __name__ == "__main__":
	raw_data = pd.read_parquet('../season-data/2022-season.parquet')
	data = ReduceToStandardSituations(raw_data)
	fields = [
		'epa',
		'air_epa',
		'yac_epa',
		'wpa',
		'air_wpa',
		'yac_wpa',
		'vegas_wpa',
		'interception',
		'fumble',
		'qb_hit',
	]
	data = ReduceToFields(data, fields)
	rolling_means = RollingMeans(data)
	print(rolling_means)
	# full_train_data = pd.DataFrame()
	# for season in range(2021, 2022):
		# print(f"fetching {season} season data")
		# raw_data = FetchPlayByPlayData(season)
		# standard_plays = ReduceToStandardSituations(raw_data)
		# fields = ['epa', 'wpa', 'interception', 'fumble']
		# standard_plays = ReduceToFields(standard_plays, fields)
		# cumulative_rates = ComputeCumulativeRatesFromRawPBP(standard_plays)
		# game_outcomes = (   # TODO: move this to its own function
			# raw_data
			# .groupby('game_id').last()
			# [['home_team', 'away_team', 'total_home_score', 'total_away_score', 'week']]
			# .query('4 <= week <=14')
		# )
		# shuffled_game_outcomes = ShuffleHomeAwayTeams(game_outcomes)
		# training_data = CreateLabeledTrainingData(shuffled_game_outcomes, cumulative_rates)
		# full_train_data = pd.concat([full_train_data, training_data])
	# print(full_train_data)
	# full_train_data.to_csv('./train.csv')
