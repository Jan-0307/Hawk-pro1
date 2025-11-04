import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

os.makedirs("models", exist_ok=True)

# Load training dataset
train_path = "data/processed/unsw/X_train.csv"
y_multi_path = "data/processed/unsw/y_train_multi_class.csv"

X_train = pd.read_csv(train_path)
y_train_multi = pd.read_csv(y_multi_path)

encoders = {}

# Encode multi-class target (attack_cat)
le_attack = LabelEncoder()
y_train_multi = le_attack.fit_transform(y_train_multi.astype(str))
joblib.dump(le_attack, "models/attack_cat_encoder.joblib")
print("Saved attack_cat encoder")

# Encode any object (string) columns in features
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    path = f"models/{col}_encoder.joblib"
    joblib.dump(le, path)
    encoders[col] = path
    print(f"Saved encoder for {col} -> {path}")

# Scale numeric features
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
scaler = MinMaxScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
joblib.dump(scaler, "models/minmax_scaler.joblib")
print("Saved MinMaxScaler")

# Save feature metadata
feature_meta = {
    "columns": X_train.columns.tolist(),
    "num_cols": num_cols,
    "cat_cols": cat_cols
}
joblib.dump(feature_meta, "models/feature_meta.joblib")
print("Saved feature metadata")