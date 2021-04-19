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



data = pd.read_csv('data1.csv')


X = train.drop(columns=['TARGET']) 
y = train['TARGET']



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