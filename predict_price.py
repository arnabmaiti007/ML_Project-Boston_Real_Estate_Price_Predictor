from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np
import pandas as pd

model = load('model.joblib') 
features = pd.DataFrame(np.array([[0.07165, 0.0, 25.65, 0, 0.581, 6.004, 84.1, 2.1974, 2, 188, 19.1, 377.67, 14.27]]))

# Pipeline
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), # For handling missing values
    ('std_scaler', StandardScaler())    # For Feature Scaling
])

features = my_pipeline.fit_transform(features)
result = model.predict(features)

print("Real Price: $%.2f"%(20.30*1000))
print("Predicted Price: $%.2f"%(result[0]*1000))