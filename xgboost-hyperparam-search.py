import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt

data = pd.read_csv('./train.csv', index_col=0)
print(np.linspace(1, 12, num=5, dtype=int))

# split our data up
train = data.copy().head(len(data)-161)
validation = data.copy().tail(161)
x = train.iloc[:,:-1]
y = train.iloc[:,-1]
print(data)

# hyperparameters
param_grid = {
	'max_depth' : np.linspace(1, 12, num=2, dtype=int),
	'gamma' : np.geomspace(0.001, 10, num=2),
	'eta' : np.geomspace(0.0001, 1, num=2),
	'lambda' : np.linspace(0.1, 2, num=2),
	'alpha' : np.linspace(0.1, 2, num=2),
	'n_estimators' : np.linspace(1, 20, num=2, dtype=int),
}

# nested cross-validation
xgb_model = xgb.XGBClassifier(
	objective="binary:logistic",
	eval_metric=brier_score_loss,
	# n_estimators = 5,
	# early_stopping_rounds = 9,
)
inner_cv = ShuffleSplit(n_splits=2, test_size=0.25)
outer_cv = ShuffleSplit(n_splits=2, test_size=0.25)
clf = GridSearchCV(
	estimator = xgb_model,
	param_grid = param_grid,
	cv = inner_cv,
	verbose = 2,
	scoring = 'neg_brier_score',
)
clf.fit(x,y)
# score = cross_val_score(
	# clf,
	# X = x,
	# y = y,
	# cv = outer_cv,
	# verbose = 2,
	# scoring='neg_brier_score',
# )
print(clf.best_score_)

# train classifier
# xgb_model = xgb.XGBClassifier(
	# objective="binary:logistic",
	# eval_metric=brier_score_loss,
	# early_stopping_rounds = 4,
	# max_depth = 4,
	# gamma = 1,
	# eta = 0.1,
	# reg_alpha = 0.8,
	# reg_lambda = 0.8
# )

