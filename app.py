import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# Kernel de Kaggle disponible sur https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
# from lightgbm_with_simple_features import *
# import lightgbm
# from lightgbm import LGBMClassifier
# from sklearn.metrics import confusion_matrix


import pickle
import re

from flask import Flask, request, jsonify, render_template

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    #print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

train_test = application_train_test(num_rows = None, nan_as_category = False)
# #bureau_balance = bureau_and_balance(num_rows = None, nan_as_category = True)
# #prev_appli = previous_applications(num_rows = None, nan_as_category = True)
# #poscash = pos_cash(num_rows = None, nan_as_category = True)
# #install_pay = installments_payments(num_rows = None, nan_as_category = True)
# #cre_card_bal = credit_card_balance(num_rows = None, nan_as_category = True)
# home_cred = pd.read_csv('HomeCredit_columns_description.csv', encoding ='cp1258')

train_test = train_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

train = train_test[train_test['TARGET'].notna()].drop(columns=['index', 'SK_ID_CURR'])
test = train_test[train_test['TARGET'].isna()].drop(columns=['index', 'SK_ID_CURR', 'TARGET'])


X = train.drop(columns=['TARGET']) 
y = train['TARGET']

X_fill = X.fillna(X.mean())
X_t = test.fillna(X.mean())

scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_fill), columns= X.columns)
# X_test = pd.DataFrame(scaler.transform(X_t), columns= test.columns)
X_train = scaler.fit_transform(X_fill)
X_test = scaler.transform(X_t)

M = pd.concat([X_train, X_test], axis=0)


app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

y_pred = model.predict_proba(M)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
# def predict():

# 	client = NumberInput(min=0, max=len(train_test))
# 	prediction = model.predict_proba(client)
# 	return render_template('index.html', prediction_text='Your Rating is: {}'.format(predict))

def predict():
    features = [int(x) for x in request.form.values()]
#     final_features = M.iloc[features]
#     prediction = model.predict_proba(final_features)
    prediction = y_pred[features]

    output = prediction

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
#	app.run(host='0.0.0.0', port=8080)
