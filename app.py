# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")

pipeline = joblib.load('pipeline.joblib')

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df
    
    input = [float(i.strip()) for i in request.form.values()]
    features = np.array([input])
    
    columns_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']
    
    df = pd.DataFrame(features, columns=columns_name)
    
    df = pipeline.transform(df)
     
    output = model.predict(df)

    return render_template('index.html', prediction_text="Predicted Price: $%.2f"%(output[0]*1000))

if __name__ == "__main__":
    app.run(debug=False)
