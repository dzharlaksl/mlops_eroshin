# data_preprocessing.py
#!/usr/bin/python3
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            x, y = np.load(os.path.join(data_dir, filename), allow_pickle=True)
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            y_scaled = scaler.transform(y)

            np.save(os.path.join(data_dir, "preprocessed_" + filename), (x_scaled, y_scaled))

if __name__ == "__main__":
    preprocess_data("train")
    preprocess_data("test")
