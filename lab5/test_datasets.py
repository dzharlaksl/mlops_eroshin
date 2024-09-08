
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np


# Функция тестирования
def test_dataset_1():
    xs = np.linspace(0, 10, 100)
    ys = 2 * xs + np.random.normal(0, 1, 100)
    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    ys_pred = model.predict(xs.reshape(-1, 1))
    mse = mean_squared_error(ys, ys_pred)
    assert mse < 5

def test_dataset_2():
    xs = np.linspace(0, 20, 100)
    ys = 1.5 * xs + np.random.normal(0, 1.5, 100)
    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    ys_pred = model.predict(xs.reshape(-1, 1))
    mse = mean_squared_error(ys, ys_pred)
    assert mse < 5

def test_dataset_3():
    xs = np.linspace(0, 15, 50) # Меньше точек
    ys = 3 * xs + np.random.normal(0, 0.5, 50)
    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    ys_pred = model.predict(xs.reshape(-1, 1))
    mse = mean_squared_error(ys, ys_pred)
    assert mse < 5

def test_dataset_noise():
    xs = np.linspace(0, 10, 100)
    ys = 2 * xs + np.random.normal(0, 1, 100)
    ys[40:60] *= 1.5
    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    ys_pred = model.predict(xs.reshape(-1, 1))
    mse = mean_squared_error(ys, ys_pred)
    assert mse < 5
    
