import pandas as pd
from sklearn.linearmodel import LinearRegression
import joblib

# Загрузка данных
traindf = pd.readcsv("train/traindataprocessed.csv")

# Тренировка модели
Xtrain = traindf[["x"]]
ytrain = traindf["y"]

model = LinearRegression()
model.fit(Xtrain, ytrain)

# Сохранение модели
joblib.dump(model, "model.joblib")
