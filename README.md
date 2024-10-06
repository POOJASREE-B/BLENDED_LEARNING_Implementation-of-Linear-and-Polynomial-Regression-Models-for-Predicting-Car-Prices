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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('car_price_prediction_.csv')
print(data.head())
X = data[['Engine Size']] 
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f'Linear Regression - Mean Squared Error: {mse_linear}')
print(f'Linear Regression - R-squared: {r2_linear}')
poly_features = PolynomialFeatures(degree=3) 
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f'Polynomial Regression - Mean Squared Error: {mse_poly}')
print(f'Polynomial Regression - R-squared: {r2_poly}')
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred_linear, color='red', label='Predicted Prices')
plt.title('Linear Regression')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred_poly, color='red', label='Predicted Prices')
plt.title('Polynomial Regression')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/4c72e532-f2c2-4793-9ccf-1ca7ef7ca4ff)
![image](https://github.com/user-attachments/assets/d443a607-9248-4f99-99d5-21bfac6d0584)
![image](https://github.com/user-attachments/assets/698cbdbe-2fd5-4289-b2c2-8a18021b02fc)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
