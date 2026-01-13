import pickle
import joblib
import json
import numpy as np
import os
import time

X_new = np.array([[6]])

# Load pickle
start = time.time()
pickle_model = pickle.load(open("model.pkl", "rb"))
pickle_time = time.time() - start

# Load joblib
start = time.time()
joblib_model = joblib.load("model.joblib")
joblib_time = time.time() - start

# Load JSON weights
with open("model_weights.json") as f:
    weights = json.load(f)

coef = np.array(weights["coef"])
intercept = np.array(weights["intercept"])

json_pred = X_new.dot(coef) + intercept

print("Pickle Prediction:", pickle_model.predict(X_new))
print("Joblib Prediction:", joblib_model.predict(X_new))
print("JSON Prediction:", json_pred)

print("\nFile Sizes:")
print("Pickle:", os.path.getsize("model.pkl"), "bytes")
print("Joblib:", os.path.getsize("model.joblib"), "bytes")
print("JSON:", os.path.getsize("model_weights.json"), "bytes")

print("\nLoad Times:")
print("Pickle:", pickle_time)
print("Joblib:", joblib_time)
