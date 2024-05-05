import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_columns = ["REGION", "YEAR", "MONTH", "SECTIONS", "WEIGHT", "WGTADJ", "STATUS", "PRICE",
                "SQFT", "BEDROOMS", "TITLED", "LOCATION", "FOUNDATION", "SECURED"]

data = pd.read_csv("Combined.csv")


# Basic statistics
print("Basic Data Summary:")
print(data.describe())

# Correlation heatmap
plt.figure(figsize=(12, 8))
plt.imshow(data.corr(), cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Matrix")
plt.xticks(np.arange(len(data_columns)), data_columns, rotation=90)
plt.yticks(np.arange(len(data_columns)), data_columns)
plt.show()