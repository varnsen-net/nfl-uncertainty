import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

data = pd.read_csv('./train.csv', index_col=0)

# split data
train = data.copy().head(len(data)-160)
validation = data.copy().tail(160)
x_train = train.iloc[:,:-1]
y_train = train.loc[:,'obj_team_win']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.01)

# fit model
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

# predict on validation set and plot calibration curve
y_pred = model.predict_proba(validation.iloc[:,:-1])
prob_pred, prob_true = calibration_curve(
	validation['obj_team_win'],
	y_pred[:,1],
	n_bins = 5,
	pos_label = 1,
	strategy = 'quantile',
)
fig, ax = plt.subplots()
ax.plot(prob_true, prob_pred)
ax.plot([0,1], [0,1])
plt.show()
