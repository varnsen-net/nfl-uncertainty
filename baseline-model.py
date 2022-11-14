import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import varnsen.plot as vplt
from joblib import dump, load

# load data
data = pd.read_csv('./train.csv', index_col=0)
pyexps = pd.concat([data['obj_pyexp'], data['adv_pyexp']]).values.reshape(-1,1)
scaled = StandardScaler().fit_transform(pyexps)
fig, ax = plt.subplot_mosaic(
    "AB",
    figsize=(5.5,2.3),
    sharey=True,
)
sns.histplot(pyexps, kde=True, legend=False, ax=ax['A'])
ax['A'].set_xlabel('Pythagorean expectation')
ax['A'].set_title('Distribution of Pythagorean expectations\nfor every team and week of NFL seasons 2002-2021')
sns.histplot(scaled, kde=True, legend=False, ax=ax['B'])
ax['B'].set_xlabel('Pythagorean expectation')
ax['B'].set_title('Distribution of z-score normalized Pythagorean expectations')
plt.tight_layout()
plt.savefig('./figures/baseline-model-feature-distributions.png')
plt.show()

# split into training and validation data
train = data.copy().iloc[:-1000]
validation = data.copy().iloc[-1000:]

# scale data
pyexps = pd.concat([train['obj_pyexp'], train['adv_pyexp']]).values.reshape(-1,1)
scaler = StandardScaler().fit(pyexps)
train['obj_pyexp'] = scaler.transform(train['obj_pyexp'].values.reshape(-1,1))
train['adv_pyexp'] = scaler.transform(train['adv_pyexp'].values.reshape(-1,1))
validation['obj_pyexp'] = scaler.transform(validation['obj_pyexp'].values.reshape(-1,1))
validation['adv_pyexp'] = scaler.transform(validation['adv_pyexp'].values.reshape(-1,1))

# model
x = train[['obj_pyexp', 'adv_pyexp', 'is_home']]
y = train['obj_team_win']
model = LogisticRegression(solver='liblinear')
cv = ShuffleSplit(n_splits=10, test_size=0.25)
scores = cross_val_score(
    model,
    x,y,
    cv=cv,
    scoring='neg_brier_score',
)

# initialize figure
mosaic = "AA;BC"
fig, ax = plt.subplot_mosaic(
    mosaic,
    figsize=(3.6,3.6),
)

# plot histogram of cross-validated scores
sns.histplot(scores, kde=True, ax=ax['A'])
ax['A'].set_xlabel('Brier score')
ax['A'].set_title('Cross-validated Brier scores')

# plot calibration curve for training data
model.fit(x,y)
y_pred = model.predict_proba(x)[:,1]
prob_pred, prob_true = calibration_curve(
    y,
    y_pred,
    n_bins = 5,
    pos_label = 1,
    strategy = 'uniform',
)
ax['B'].plot(prob_true, prob_pred)
ax['B'].plot([0,1], [0,1])
ax['B'].set_xlabel('True probability')
ax['B'].set_ylabel('Predicted probability')
ax['B'].set_title('Calibration curve (training data)')

# plot calibration curve for validation data
x_val = validation[['obj_pyexp', 'adv_pyexp', 'is_home']]
y_val = validation['obj_team_win']
y_pred = model.predict_proba(x_val)[:,1]
prob_pred, prob_true = calibration_curve(
    y_val,
    y_pred,
    n_bins = 5,
    pos_label = 1,
    strategy = 'uniform',
)
ax['C'].plot(prob_true, prob_pred)
ax['C'].plot([0,1], [0,1])
ax['C'].set_xlabel('True probability')
ax['C'].set_title('Calibration curve (validation data)')

# plt.tight_layout()
# plt.savefig('./figures/baseline-model-validation.png')

# retrain on all data and save model
# x = data[['obj_pyexp', 'adv_pyexp', 'is_home']].copy()
# y = data['obj_team_win'].copy()
# x['obj_pyexp'] = scaler.transform(x['obj_pyexp'].values.reshape(-1,1))
# x['adv_pyexp'] = scaler.transform(x['adv_pyexp'].values.reshape(-1,1))
# model = LogisticRegression(solver='liblinear')
# model.fit(x,y)
# dump(model, './models/baseline-logistic-regression.joblib')
# dump(scaler, './models/baseline-scaler.joblib')
