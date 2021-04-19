import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Kernel de Kaggle disponible sur https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
from lightgbm_with_simple_features import *
import lightgbm
from lightgbm import LGBMClassifier

# import pickle
import re

train_test = application_train_test(num_rows = None, nan_as_category = False)

train_test = train_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

train = train_test[train_test['TARGET'].notna()].drop(columns=['index', 'SK_ID_CURR'])

X = train.drop(columns=['TARGET']) 
y = train['TARGET']

X_fill = X.fillna(X.mean())

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_fill), columns= X.columns)

model = LGBMClassifier(max_depth=24, n_estimators=836, num_leaves=23,
                     learning_rate=0.02,
                     min_child_weight= 95.7889530150502,
                     min_split_gain= 0.5331652849730171,
                     reg_alpha= 0.6918771139504734,
                     reg_lambda= 0.31551563100606295,
                     colsample_bytree= 0.20445224973151743,
                     subsample= 0.8781174363909454, 
                     is_unbalance=True, random_state=1, force_row_wise=True, objective='binary')


model.fit(X_train, y)
y_pred = model.predict_proba(X_train)
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_pred)