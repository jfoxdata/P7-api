from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import pandas as pd
import numpy as np



# Kernel de Kaggle disponible sur https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
# from lightgbm_with_simple_features import *
# import lightgbm
# from lightgbm import LGBMClassifier
# from sklearn.metrics import confusion_matrix


import pickle
import re

from sklearn.preprocessing import StandardScaler






app=Flask(__name__)

# data = pd.read_csv('data1.csv', index_col=0)


# X = data.iloc[:307507]
# X_test = data.iloc[307508:]

# scaler = StandardScaler()
# scaler.fit(X)
# X_s = scaler.transform(data)

# M = X_s

model = pickle.load(open('model.pkl', 'rb'))

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
