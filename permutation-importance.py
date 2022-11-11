import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

data = pd.read_csv('./train.csv', index_col=0)
param_grid = pd.read_csv('./grid-search.csv', index_col=0)
params = param_grid.query('rank_test_score == 1')


# split our data up
train = data.copy().head(len(data)-161)
validation = data.copy().tail(161)
x = train.iloc[:,:-1]
y = train.iloc[:,-1]
print(data)

# hyperparams
# note that the scikit implementation does not accept **kwargs, so
# we're stuck with an ugly list of variables
subsample = params['param_subsample'].values[0]
n_estimators = params['param_n_estimators'].values[0]
min_child_weight = params['param_min_child_weight'].values[0]
max_depth = params['param_max_depth'].values[0]
reg_lambda = params['param_lambda'].values[0]
gamma = params['param_gamma'].values[0]
eta = params['param_eta'].values[0]
alpha = params['param_alpha'].values[0]

# initialize and fit model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric=brier_score_loss,
    tree_method = 'gpu_hist',
    gpu_id = 0,
    subsample = subsample,
    n_estimators = n_estimators,
    min_child_weight = min_child_weight,
    max_depth = max_depth,
    reg_lambda = reg_lambda,
    gamma = gamma,
    eta = eta,
    alpha = alpha,
)
xgb_model.fit(x,y)

# permutation importance
result = permutation_importance(
    xgb_model, validation.iloc[:,:-1], validation.iloc[:,-1], n_repeats=250, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=x.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()
