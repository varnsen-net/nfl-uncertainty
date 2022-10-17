import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import matplotlib.pyplot as plt

data = pd.read_csv('./train.csv', index_col=0)

# split our data up
train = data.copy().head(len(data)-161)
validation = data.copy().tail(161)
x_train = train.iloc[:,:-1]
y_train = train.loc[:,'obj_team_win']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# train classifier
xgb_model = xgb.XGBClassifier(
	objective="binary:logistic",
	eval_metric=brier_score_loss,
	early_stopping_rounds = 4,
	max_depth = 4,
	gamma = 1,
	eta = 0.1,
	reg_alpha = 0.8,
	reg_lambda = 0.8
)
xgb_model.fit(
	x_train, y_train,
	eval_set=[(x_test, y_test)],
)

# calibrate probabilities
calibrated_model = CalibratedClassifierCV(xgb_model, cv='prefit')
calibrated_model.fit(
	validation.iloc[:,:-1], validation['obj_team_win'],
)

# predict on validation set and plot calibration curve
y_pred = calibrated_model.predict_proba(validation.iloc[:,:-1])
prob_pred, prob_true = calibration_curve(
	validation['obj_team_win'],
	y_pred[:,1],
	n_bins = 6,
	pos_label = 1,
	strategy = 'quantile',
)
fig, ax = plt.subplots()
ax.plot(prob_true, prob_pred)
ax.plot([0.2,0.8], [0.2,0.8])
plt.show()


