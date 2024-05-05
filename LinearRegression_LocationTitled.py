import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

data_columns = ["REGION", "YEAR", "MONTH", "SECTIONS", "WEIGHT", "WGTADJ", "STATUS", "PRICE",
                "SQFT", "BEDROOMS", "TITLED", "LOCATION", "FOUNDATION", "SECURED"]

data = pd.read_csv("PDABUAN6340\\Project\\Combined.csv")

target_var = "TITLED"
predictor_var = ["LOCATION"]

# Train-test split (manually)
train_size = int(0.8 * len(data))
X_train = data[predictor_var][:train_size]
X_test = data[predictor_var][train_size:]
y_train = data[target_var][:train_size]
y_test = data[target_var][train_size:]

# linear regression one predictor for simplicitys
slope, intercept, r_value, p_value, std_err = stats.linregress(X_train["LOCATION"], y_train)

# Linear regression
print("\nRegression Metrics:")
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value ** 2}")
print(f"P-value: {p_value}")
print(f"Standard Error: {std_err}")

# Prediction
y_pred = intercept + slope * X_test["LOCATION"]

# Calculate error metrics
mse = np.mean((y_pred - y_test) ** 2)
mae = np.mean(np.abs(y_pred - y_test))
r2 = r_value ** 2
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test["LOCATION"], y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test["LOCATION"], y_pred, color='red', label='Regression Line')
plt.title("Scatter Plot with Regression Line")
plt.xlabel("LOCATION")
plt.ylabel("TITLED")
plt.legend()
plt.show()
