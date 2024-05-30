# model_testing.py
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка данных и модели
test_df = pd.read_csv("test/test_data_processed.csv")
model = joblib.load("model.joblib")

# Тестирование модели
X_test = test_df[["x"]]
y_test = test_df["y"]

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
