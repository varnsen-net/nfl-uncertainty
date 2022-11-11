import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV, cross_val_score
import xgboost as xgb


data = pd.read_csv('./train.csv', index_col=0)
print(np.geomspace(0.0001, 1, num=10))

# split our data up
train = data.copy().head(len(data)-161)
validation = data.copy().tail(161)
x = train.iloc[:,:-1]
y = train.iloc[:,-1]
print(data)

# hyperparameters
param_grid = {
    'max_depth' : np.linspace(2, 10, num=9, dtype=int),
    'gamma' : np.linspace(0, 10, num=22),
    'eta' : np.geomspace(0.0001, 1, num=100),
    'lambda' : np.linspace(0.1, 2, num=100),
    'alpha' : np.linspace(0.1, 2, num=100),
    'min_child_weight' : np.linspace(0, 10, num=11, dtype=int),
    'subsample' : np.linspace(0.5, 1, num=11),
    'n_estimators' : np.linspace(500, 1500, num=11, dtype=int),
}

# nested cross-validation
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric=brier_score_loss,
    tree_method = 'gpu_hist',
    gpu_id = 0,
    # n_estimators = 1000,
    # early_stopping_rounds = 9,
)
inner_cv = ShuffleSplit(n_splits=5, test_size=0.25)
outer_cv = ShuffleSplit(n_splits=5, test_size=0.25)
clf = RandomizedSearchCV(
    estimator = xgb_model,
    param_distributions = param_grid,
    n_iter = 50,
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
print(clf.best_estimator_)
pd.DataFrame(clf.cv_results_).to_csv('./grid-search.csv')

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

