# Task 3.4 : Model Persistence - Saving Models with Different Libraries
import pickle
import joblib
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Create data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([3,5,7,9,11])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save using pickle
pickle.dump(model, open("model.pkl", "wb"))

# Save using joblib
joblib.dump(model, "model.joblib")

# Save weights as JSON
weights = {
    "coef": model.coef_.tolist(),
    "intercept": model.intercept_.tolist()
}

with open("model_weights.json", "w") as f:
    json.dump(weights, f)

print("Models saved successfully")
