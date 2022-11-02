import pandas as pd
import varnsen.tables as vtb


for year in range(2022,2023):
	season_data = vtb.FetchPlayByPlayData(year)
	path = f"./season-data/{year}-season.parquet"
	season_data.to_parquet(
		path,
		index=False,
	)
	print(f"Success! Saved to {path}")
