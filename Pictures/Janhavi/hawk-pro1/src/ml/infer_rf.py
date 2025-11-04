import joblib
import pandas as pd
import numpy as np
import os

model = joblib.load("models/random_forest_model.joblib")
meta = joblib.load("models/feature_meta.joblib")  # contains 'columns' used during training

# load a sample or test set
X_test = pd.read_csv("data/processed/unsw/X_test.csv")
X_test.columns = meta['columns']

# predict
pred = model.predict(X_test)
probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

print("Predictions (first 5):", pred[:5])
if probs is not None:
    print("Probabilities (first 5):", np.round(probs[:5], 3))
# save sample predictions
pd.DataFrame({"pred": pred}).to_csv("models/predictions_sample.csv", index=False)
print("Saved sample predictions -> models/predictions_sample.csv")

# Feature importance
imp = getattr(model, "feature_importances_", None)
if imp is not None:
    print("Top features (idx,score):", list(zip(np.argsort(imp)[-10:][::-1], sorted(imp)[-10:])))
else:
    print("Model has no feature_importances_")