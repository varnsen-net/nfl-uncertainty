import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('./train.csv', index_col=0)

train = data.copy().head(len(data)-250)
validation = data.copy().tail(250)
print(validation)

x_train = train.iloc[:,:-1]
y_train = train.loc[:,'obj_team_win']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
y_pred = model.predict_proba(validation.iloc[:,:-1])
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
validation['model_proba'] = pd.cut(y_pred[:,1], bins=bins)
results = validation[['obj_team_win', 'model_proba']]
grouped = results.groupby('model_proba').agg(['count', 'sum'])['obj_team_win']
grouped['win_rate_actual'] = grouped['sum'] / grouped['count']
print(grouped)

