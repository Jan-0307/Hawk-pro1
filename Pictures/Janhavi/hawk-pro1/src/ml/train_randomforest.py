import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("models", exist_ok=True)

# 1) Load processed data (prefer data/processed/unsw, fallback to local x_train/x_test)
data_dir = os.path.join("data", "processed", "unsw")
candidates = [
    os.path.join(data_dir, "x_train.csv"),
    os.path.join(data_dir, "x_train.csv"),
    "x_train.csv"
]
x_train_path = next((p for p in candidates if os.path.exists(p)), None)
if not x_train_path:
    raise FileNotFoundError(
        "x_train not found. Place X_train.csv/x_train.csv in data/processed/unsw/ or project root."
    )
x_train = pd.read_csv(x_train_path)

candidates = [
    os.path.join(data_dir, "X_test.csv"),
    os.path.join(data_dir, "x_test.csv"),
    "x_test.csv"
]
x_test_path = next((p for p in candidates if os.path.exists(p)), None)
if not x_test_path:
    raise FileNotFoundError(
        "x_test not found. Place X_test.csv/x_test.csv in data/processed/unsw/ or project root."
    )
x_test = pd.read_csv(x_test_path)

if os.path.exists(os.path.join(data_dir, "y_train_multi_class.csv")) and os.path.exists(os.path.join(data_dir, "y_test_multi_class.csv")):
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_multi_class.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test_multi_class.csv")).values.ravel()
    print("Using multi-class labels")
elif os.path.exists(os.path.join(data_dir, "y_train_binary.csv")) and os.path.exists(os.path.join(data_dir, "y_test_binary.csv")):
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_binary.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test_binary.csv")).values.ravel()
    print("Using binary labels")
elif os.path.exists(os.path.join(data_dir, "y_train.csv")) and os.path.exists(os.path.join(data_dir, "y_test.csv")):
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    print("Using default y_train.csv/y_test.csv")
else:
    # fallback to local names
    if os.path.exists("y_train_multi_class.csv") and os.path.exists("y_test_multi_class.csv"):
        y_train = pd.read_csv("y_train_multi_class.csv").values.ravel()
        y_test  = pd.read_csv("y_test_multi_class.csv").values.ravel()
        print("Using multi-class labels (local)")
    elif os.path.exists("y_train_binary.csv") and os.path.exists("y_test_binary.csv"):
        y_train = pd.read_csv("y_train_binary.csv").values.ravel()
        y_test  = pd.read_csv("y_test_binary.csv").values.ravel()
        print("Using binary labels (local)")
    elif os.path.exists("y_train.csv") and os.path.exists("y_test.csv"):
        y_train = pd.read_csv("y_train.csv").values.ravel()
        y_test  = pd.read_csv("y_test.csv").values.ravel()
        print("Using default y_train.csv/y_test.csv (local)")
    else:
        raise FileNotFoundError("No label files found. Place y_train_* and y_test_* in data/processed/unsw/ or project root.")

# 2) Load preprocessing tools (only feature_meta is needed)
meta = joblib.load("models/feature_meta.joblib")

# Assign column names to X_train and X_test based on feature_meta
# This is important for consistency even if not strictly needed by RandomForestClassifier
x_train.columns = meta['columns']
x_test.columns = meta['columns']

# The data in X_train and X_test CSVs is already scaled, so no need to apply scaler.transform again.

# 3) Train Random Forest
SEED = 42
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(x_train, y_train)

# 4) Save model
out_path = "models/random_forest_model.joblib"
joblib.dump(rf, out_path)
print("✅ RandomForest saved ->", out_path)

# 5) Evaluate
y_pred = rf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))