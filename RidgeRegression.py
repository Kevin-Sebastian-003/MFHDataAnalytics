import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import solve

data_columns = ["REGION", "YEAR", "MONTH", "SECTIONS", "WEIGHT", "WGTADJ", "STATUS", "PRICE",
                "SQFT", "BEDROOMS", "TITLED", "LOCATION", "FOUNDATION", "SECURED"]

data = pd.read_csv("Combined.csv")
target_var = "PRICE"
predictor_vars = ["REGION"]


X = data[predictor_vars].values
y = data[target_var].values
X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
alpha = 1.0  # Regularization strength

# Ridge regression
I = np.eye(X_with_bias.shape[1])
ridge_matrix = np.dot(X_with_bias.T, X_with_bias) + alpha * I
ridge_rhs = np.dot(X_with_bias.T, y)
coefficients = solve(ridge_matrix, ridge_rhs)
intercept = coefficients[0]
slopes = coefficients[1:]

# predictions
y_pred = intercept + np.dot(X, slopes)

# metrics
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print("Ridge Regression Results:")
print(f"Intercept: {intercept}")
for var, slope in zip(predictor_vars, slopes):
    print(f"Slope for {var}: {slope}")
print("\nRegression Metrics:")
print(f"R-squared (R²): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(data[predictor_vars[0]], data[target_var], color='blue', alpha=0.5, label='Actual Data')
plt.plot(data[predictor_vars[0]], y_pred, color='red', label='Ridge Regression Line')
plt.xlabel(predictor_vars[0])
plt.ylabel(target_var)
plt.title("Ridge Regression")
plt.legend()
plt.show()

# Residual plot
residuals = y - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='green', label='Residuals')
plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at zero residual
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.legend()
plt.show()