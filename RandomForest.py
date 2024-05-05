import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_columns = ["REGION", "YEAR", "MONTH", "SECTIONS", "WEIGHT", "WGTADJ", "STATUS", "PRICE",
                "SQFT", "BEDROOMS", "TITLED", "LOCATION", "FOUNDATION", "SECURED"]
data = pd.read_csv("Combined.csv")

target_var = "PRICE"  # Example target variable
predictor_vars = ["REGION", "SQFT", "SECTIONS", "WEIGHT", "YEAR"]

# Split
X_train, X_test, y_train, y_test = train_test_split(data[predictor_vars], data[target_var], test_size=0.2, random_state=42)

# train Random Forest model
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # 100 trees, max depth 10
random_forest.fit(X_train, y_train)

# predictions
y_pred = random_forest.predict(X_test)

# regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Random Forest Regression Results:")
print(f"R-squared (R²): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', linewidth=2)  # Identity line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression")
plt.legend()
plt.show()

# Residual plot to check for patterns or non-linearity
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='green', alpha=0.5, label='Residuals')
plt.axhline(0, color='red', linestyle='--', linewidth=2)  # Line at zero residual
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.legend()
plt.show()


feature_importances = random_forest.feature_importances_

# variable ranking
importance_df = pd.DataFrame({"Feature": predictor_vars, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)  # Sort by importance
print("Feature Importances:")
print(importance_df)

# feature importances bar plot
plt.figure(figsize=(10, 6))
plt.bar(importance_df["Feature"], importance_df["Importance"], color='skyblue')
plt.xlabel("Predictor Variables")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest")
plt.show()