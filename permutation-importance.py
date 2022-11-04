import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

train = pd.read_csv('./train.csv', index_col='game_id')
x = train.iloc[:,:-1]
y = train.iloc[:,-1]

# xgboost 
rf = RandomForestClassifier().fit(x_train, y_train)

# permutation importance
# result = permutation_importance(
	# rf, x_test, y_test, n_repeats=40, random_state=42, n_jobs=2
# )

# sorted_importances_idx = result.importances_mean.argsort()
# importances = pd.DataFrame(
	# result.importances[sorted_importances_idx].T,
	# columns=x.columns[sorted_importances_idx],
# )
# ax = importances.plot.box(vert=False, whis=10)
# ax.set_title("Permutation Importances (test set)")
# ax.axvline(x=0, color="k", linestyle="--")
# ax.set_xlabel("Decrease in accuracy score")
# ax.figure.tight_layout()
# plt.show()
