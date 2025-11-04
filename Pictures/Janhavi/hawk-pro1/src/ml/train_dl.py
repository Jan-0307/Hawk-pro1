#!/usr/bin/env python3

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ---- reproducibility (best-effort) ----
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path} (cwd={os.getcwd()})")
    return pd.read_csv(path)

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_or_create_label_encoder(y_train_series, encoder_path):
    ensure_dir(os.path.dirname(encoder_path) or ".")
    if os.path.exists(encoder_path):
        le = joblib.load(encoder_path)
        print(f"Loaded label encoder: {encoder_path}")
    else:
        le = LabelEncoder()
        le.fit(y_train_series.astype(str))
        joblib.dump(le, encoder_path)
        print(f"Created and saved label encoder -> {encoder_path}")
    return le

def prepare_data(xtrain_path, xtest_path, ytrain_path, ytest_path, scaler_path, encoder_path):
    print("Loading data...")
    X_train = load_csv(xtrain_path)
    X_test  = load_csv(xtest_path)
    y_train = load_csv(ytrain_path)
    y_test  = load_csv(ytest_path)

    # if y files are single-column CSVs with no header, handle that:
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:,0]
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test = y_test.iloc[:,0]

    # load scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run preprocessing to create scaler.")
    scaler = joblib.load(scaler_path)
    print("Loaded scaler:", scaler_path)

    # detect numeric columns and scale them
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in X_train; check your processed CSV.")
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # load or create label encoder for multi-class
    le = load_or_create_label_encoder(y_train, encoder_path)

    y_train_enc = le.transform(y_train.astype(str))
    y_test_enc  = le.transform(y_test.astype(str))

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Convert X to numpy arrays (2D: samples x features)
    X_train_np = X_train.values.astype('float32')
    X_test_np  = X_test.values.astype('float32')

    print("Prepared data shapes:",
          "X_train:", X_train_np.shape, "y_train:", y_train_enc.shape,
          "X_test:", X_test_np.shape, "y_test:", y_test_enc.shape)

    meta = {
        "num_features": X_train_np.shape[1],
        "num_classes": num_classes,
        "label_encoder": encoder_path
    }

    return X_train_np, X_test_np, y_train_enc, y_test_enc, le, meta

def train_model(X_train, X_test, y_train, y_test, meta, out_path,
                epochs=20, batch_size=128, lr=1e-3, patience=5):
    """
    Train a scikit-learn RandomForest classifier as a replacement for the TF model.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    print("Training RandomForestClassifier...")

    model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    # save model
    joblib.dump(model, out_path)
    print("Saved model ->", out_path)

    # evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test accuracy: {acc:.4f}  macro-F1: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    metrics = {"accuracy": float(acc), "f1_macro": float(f1)}
    return model, metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xtrain", required=True)
    p.add_argument("--xtest", required=True)
    p.add_argument("--ytrain", required=True)
    p.add_argument("--ytest", required=True)
    p.add_argument("--scaler", default="models/minmax_scaler.joblib")
    p.add_argument("--encoder", default="models/attack_cat_encoder.joblib")
    p.add_argument("--out", default="models/best_model_rf.joblib")
    p.add_argument("--epochs", type=int, default=20)   # kept for CLI compatibility (ignored)
    p.add_argument("--batch", type=int, default=128)   # ignored
    p.add_argument("--lr", type=float, default=1e-3)    # ignored
    p.add_argument("--patience", type=int, default=6)  # ignored
    return p.parse_args()

def main():
    args = parse_args()

    X_train, X_test, y_train, y_test, label_encoder, meta = prepare_data(
        args.xtrain, args.xtest, args.ytrain, args.ytest, args.scaler, args.encoder)

    print("Starting training...")
    model, metrics = train_model(X_train, X_test, y_train, y_test, meta, args.out,
                                epochs=args.epochs, batch_size=args.batch, lr=args.lr, patience=args.patience)

    print("Training complete. Saved model ->", args.out)
    print("Label classes:", list(label_encoder.classes_))
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()