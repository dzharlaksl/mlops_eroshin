# model_preparation.py
#!/usr/bin/python3
import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

def train_model(data_dir, model_filename):
    X_train, y_train = [], []

    for filename in os.listdir(data_dir):
        if filename.startswith("preprocessed_train_data"):
            x, y = np.load(os.path.join(data_dir, filename), allow_pickle=True)
            X_train.append(x)
            y_train.append(y)

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model("train", "model.pkl")
