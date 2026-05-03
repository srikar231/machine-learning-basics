import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X = np.array([
    [2, 7],
    [4, 6],
    [6, 8],
    [8, 7],
    [3, 6],
    [5, 8],
    [7, 7],
    [9, 8]
])

y = np.array([30, 45, 70, 80, 35, 60, 75, 90])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("X Train:", X_train)
print("X Test:", X_test)
print("y Train:", y_train)
print("y Test:", y_test)

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)
print("Prediction:", prediction)

mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediction)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

print("Prediction for [5,7]:", model.predict([[5, 7]]))
