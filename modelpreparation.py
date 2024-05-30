import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Загрузка данных
traindf = pd.read_csv("train/train_data_processed.csv")

# Тренировка модели
Xtrain = traindf[["x"]]
ytrain = traindf["y"]

model = LinearRegression()
model.fit(Xtrain, ytrain)

# Сохранение модели
joblib.dump(model, "model.joblib")
