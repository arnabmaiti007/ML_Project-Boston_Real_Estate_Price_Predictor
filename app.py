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
    
    print(features)
    print(10)
    
    columns_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']
    
    print(100)
    
    df = pd.DataFrame(features, columns=columns_name)
    
    print(200)
    
    #my_pipeline = Pipeline([
    #    ('imputer', SimpleImputer(strategy="median")), # For handling missing values
    #    ('std_scaler', StandardScaler())    # For Feature Scaling
    #])
    print(df)
    
    
    df = pipeline.transform(df)
    
    print(df)
    
    output = model.predict(df)
    
    print(output)
    print(500)
    return render_template('index.html', prediction_text="Predicted Price: $%.2f"%(output[0]*1000))

if __name__ == "__main__":
    app.run()