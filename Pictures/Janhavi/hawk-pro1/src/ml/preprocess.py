import pandas as pd

train_data = pd.read_csv("unsw_train.csv")
test_data = pd.read_csv("unsw_test.csv")

def preprocess_data(df):
    print("train shape:", train_data.shape)
    print("test shape:", test_data.shape)
    print(train_data.head())