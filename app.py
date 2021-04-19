from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
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








app=Flask(__name__)

# data = pd.read_csv('data1.csv', index_col=0)


# X = data.iloc[:307507]
# X_test = data.iloc[307508:]


# scaler = StandardScaler()
# # X_train = pd.DataFrame(scaler.fit_transform(X_fill), columns= X.columns)
# # X_test = pd.DataFrame(scaler.transform(X_t), columns= test.columns)
# X_train = scaler.fit_transform(X)
# X_s = scaler.transform(X_test)

# M = np.concatenate((X_train, X_s))

# model = pickle.load(open('model.pkl', 'rb'))

# y_pred = model.predict_proba(M)

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
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
