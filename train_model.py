# ========== Real Estate Price Predictor ==========
#          [ Machine Learning Project ]


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# Loading the DATASET
housing = pd.read_csv("data.csv")

# Spliting the dataset to Train Data and Test data
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)  # n_splits = 1 means it will only iterate(i.e- split) for 1 time only.
for train_index, test_index in sss.split(housing, housing['CHAS']):   # housing['CHAS'] - provided variable.
    #print(train_index,test_index)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Pipeline
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), # For handling missing values
    ('std_scaler', StandardScaler())    # For Feature Scaling
])

# Creating inputs for the model
features = strat_train_set.drop('MEDV', axis=1)
labels = strat_train_set['MEDV']

# Preparing the data
features_prepared = my_pipeline.fit_transform(features)

# Selecting the Model:
model = RandomForestRegressor()

# training the prepared data:
model.fit(features_prepared, labels)

# Creating Class and function for evaluation:
class Evaluate:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels

    def rmse(self):
        from sklearn.metrics import mean_squared_error
        prediction = self.model.predict(self.features)
        mse = mean_squared_error(self.labels, prediction)
        rmse = np.sqrt(mse)
        print("-- RMSE --")
        print(f"Root Mean Square Error is: {rmse}")

    def cross_validation(self):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.features, self.labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)  # as scores is negative values
        print("-- Cross Validation --")
        #print("Scores:", rmse_scores)
        print("Mean: ", rmse_scores.mean())
        print("Standard deviation: ", rmse_scores.std())

print(f"Trained examples count: {len(features_prepared)}")        

# Evaluating on the Training data:
e = Evaluate(model, features_prepared, labels)
e.rmse()
e.cross_validation()

# Saving the model:
dump(model, 'model.joblib')

print("Model Saved.")
print("----- Program Finished -----")