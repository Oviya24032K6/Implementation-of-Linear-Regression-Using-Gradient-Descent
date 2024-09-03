# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph

## Program:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X, y, learning_rate=0.1, num_iters=1000):
    # Add a column of ones to X for the intercept term (bias)
    X = np.c_[np.ones(len(X)), X]
    # Initialize theta (model parameters) with zeros
    theta = np.zeros((X.shape[1], 1))
    
    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta)
        # Calculate errors
        errors = predictions - y
        # Update theta using gradient descent
        theta -= learning_rate * (1/len(X)) * X.T.dot(errors)
    
    return theta

# Load the dataset
data = pd.read_csv('50_Startups.csv')

# Display the first five rows of the dataset
print("First five rows of the dataset:")
print(data.head())

# Extract features (excluding the 'State' column) and target variable 'Profit'
X = data.iloc[:, :-1].select_dtypes(include=[np.number]).values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Scaling the features and the target variable
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train the linear regression model using gradient descent
theta = linear_regression(X_scaled, y_scaled)

# Predict profit for a new city (example data point)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
new_data_scaled = scaler_X.transform(new_data)
prediction_scaled = np.dot(np.append(1, new_data_scaled), theta).reshape(-1, 1)

# Inverse transform the prediction to get the original scale
predicted_profit = scaler_y.inverse_transform(prediction_scaled)

# Output the prediction
print(f"\nPredicted Profit for the city: {predicted_profit[0][0]:.2f}")

```

## Output:
![image](https://github.com/user-attachments/assets/588fbcc9-af1c-43c6-ac5f-4da28ccf1aba)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
