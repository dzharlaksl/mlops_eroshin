# model_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Загрузка данных
train_df = pd.read_csv("train/train_data.csv")
test_df = pd.read_csv("test/test_data.csv")

# Предобработка данных
scaler = StandardScaler()
train_df[["x", "y"]] = scaler.fit_transform(train_df[["x", "y"]])
test_df[["x", "y"]] = scaler.transform(test_df[["x", "y"]])

# Сохранение предобработанных данных
train_df.to_csv("train/train_data_processed.csv", index=False)
test_df.to_csv("test/test_data_processed.csv", index=False)
