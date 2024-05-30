import numpy as np
import pandas as pd
import os

# Создание директорий train и test
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)


# Функция для создания набора данных
def createdataset(numsamples, noise=False):
    np.random.seed(42)
    x = np.linspace(0, 100, numsamples)
    y = 2 * x + 1 + (np.random.normal(0, 25, numsamples) if noise else 0)
    return pd.DataFrame({"x": x, "y": y})


# Создание и сохранение данных
traindf = createdataset(1000, noise=True)
testdfnormal = createdataset(300)
testdfanomalies = createdataset(300, noise=True)

traindf.to_csv("train/train_data.csv", index=False)
testdfnormal.to_csv("test/test_data.csv", index=False)
testdfanomalies.to_csv("test/test_data_with_noise.csv", index=False)
