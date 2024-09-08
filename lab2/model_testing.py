# model_testing.py
#!/usr/bin/python3
import os
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

def test_model(data_dir, model_filename):
    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = [], []

    for filename in os.listdir(data_dir):
        if filename.startswith("preprocessed_test_data"):
            x, y = np.load(os.path.join(data_dir, filename), allow_pickle=True)
            X_test.append(x)
            y_test.append(y)

    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print("Model test MSE is:", mse)

if __name__ == "__main__":
    test_model("test", "model.pkl")
