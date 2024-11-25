# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset
2. Split the data into training and testing dataset
3. Train a Linear Regression Model
4. Train a Polynomial Regression model
5. Visualize the result
## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: POOJASREE B
RegisterNumber:  212223040148
*/
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Select features and target variable
X = data[['enginesize']]  # Predictor
y = data['price']         # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate models
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression R^2 score:", r2_score(y_test, y_pred_linear))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression R^2 score:", r2_score(y_test, y_pred_poly))

# Visualization: Linear Regression
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_test, y_pred_linear, color='blue', label='Linear')
plt.title('Linear Regression')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

# Visualization: Polynomial Regression
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_test, y_pred_poly, color='green', label='Polynomial')
plt.title('Polynomial Regression')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/9619676a-8f31-41f6-956f-97762202bf7f)

![image](https://github.com/user-attachments/assets/38ee892d-85e9-47f3-828a-b8763b8574ef)

![image](https://github.com/user-attachments/assets/d901838b-baed-4db3-bd46-484ac7b52e69)


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
