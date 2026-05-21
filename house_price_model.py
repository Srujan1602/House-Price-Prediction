import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("house_data.csv")

X = data[["area", "bedrooms", "bathrooms", "floors"]].values
y = data["price"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

X_train_matrix = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_matrix = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

theta = np.linalg.pinv(X_train_matrix.T @ X_train_matrix) @ X_train_matrix.T @ y_train

y_pred = X_test_matrix @ theta

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("House Price Prediction Model")
print("----------------------------")
print("Model trained successfully")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

print("\nEnter house details")

area = float(input("Area in sqft: "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
floors = int(input("Floors: "))

new_house = np.array([[area, bedrooms, bathrooms, floors]])

new_house_scaled = (new_house - mean) / std
new_house_matrix = np.c_[np.ones(new_house_scaled.shape[0]), new_house_scaled]

predicted_price = new_house_matrix @ theta

print("\nPredicted House Price: ₹", round(predicted_price[0][0], 2))