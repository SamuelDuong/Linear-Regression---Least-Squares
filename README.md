# Linear Regression - Least Squares


## Project Overview

This project involves coding a version of least squares regression in Python. The assignment includes:

- Calculating least squares weights
- Reading data into Pandas DataFrame
- Selecting data by column
- Implementing column cutoffs

### Expected Time

- **2 hours**

### Motivation

Least squares regression offers a way to build a closed-form and interpretable model.

### Objectives

- Test Python and Pandas competency
- Ensure understanding of the mathematical foundations behind least squares regression

### Problem

Using housing data, we will attempt to predict house prices using the living area with a regression model.

### Data

Our data comes from Kaggle's House Prices Dataset. For more details on the data, refer to the Kaggle [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

## Introduction and Review

Linear regression using least squares is solvable exactly, without requiring approximation, provided certain basic assumptions are met. This project will use the matrix version of the least squares solution to derive the desired result:

$$
w _{LS} = (X^T X)^{−1}X^T y
$$

Where:
- \( w_{LS} \) is the vector of weights we are trying to find.
- \( X \) is the matrix of inputs.
- \( y \) is the output vector.

In the equation, \( X \) is defined to have a vector of 1 values as its first column. For example, if there is one input value per data point, \( X \) takes the form:

$$
X = \begin{bmatrix}
1 & x_{11} \\
1 & x_{21} \\
\vdots & \vdots \\
1 & x_{n1}
\end{bmatrix}
$$

## Data Exploration

Before coding the algorithm, we will explore our data using Python's Pandas and visualize it with Matplotlib.

### Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
### Visualizations
```python
Y = data['SalePrice']
X = data['GrLivArea']
plt.scatter(X, Y, marker = "x")
plt.title("Sales Price vs. Living Area (excl. basement)")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

data.plot('YearBuilt', 'SalePrice', kind='scatter', marker='x')

```python

def inverse_of_matrix(mat):
    """Calculate and return the multiplicative inverse of a matrix."""
    return np.linalg.inv(mat)
```
```python
def read_to_df(file_path):
    """Read on-disk data and return a dataframe."""
    return pd.read_csv(file_path)
```

```python
def select_columns(data_frame, column_names):
    """Return a subset of a dataframe by column names."""
    return data_frame[column_names]
```
### 4. Subsetting Data by Value
```python
def column_cutoff(data_frame, cutoffs):
    """Subset data frame by cutting off limits on column values."""
    for column_limits in cutoffs:
        data_frame = data_frame.loc[(data_frame[column_limits[0]] >= column_limits[1]) & 
                                    (data_frame[column_limits[0]] <= column_limits[2])]
    return data_frame
```
### 5. Calculating Least Squares Weights
```python
def least_squares_weights(input_x, target_y):
    """Calculate linear regression least squares weights."""
    if input_x.shape[0] < input_x.shape[1]:
        input_x = np.transpose(input_x)
    if target_y.shape[0] < target_y.shape[1]:
        target_y = np.transpose(target_y)
    ones = np.ones((len(target_y), 1))
    augmented_x = np.concatenate((ones, input_x), axis=1)
    left_multiplier = np.matmul(np.linalg.inv(np.matmul(np.transpose(augmented_x), augmented_x)), np.transpose(augmented_x))
    w_ls = np.matmul(left_multiplier, target_y)
    return w_ls
```
### Testing Function
```python
print("test", inverse_of_matrix([[1, 2], [3, 4]]), "\n")
print("From Data:\n", inverse_of_matrix(data.iloc[:2,:2]))
```
### Testing on Real Data
```python
df = read_to_df(tr_path)
df_sub = select_columns(df, ['SalePrice', 'GrLivArea', 'YearBuilt'])
cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
df_sub_cutoff = column_cutoff(df_sub, cutoffs)
X = df_sub_cutoff['GrLivArea'].values
Y = df_sub_cutoff['SalePrice'].values

# Reshaping for input into function
training_y = np.array([Y])
training_x = np.array([X])

weights = least_squares_weights(training_x, training_y)

print(weights)

max_X = np.max(X) + 500
min_X = np.min(X) - 500

# Choose points evenly spaced between min_x and max_x
reg_x = np.linspace(min_X, max_X, 1000)

# Use the equation for our line to calculate y values
reg_y = weights[0][0] + weights[1][0] * reg_x

plt.plot(reg_x, reg_y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='k', label='Data')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()
```
### Calculating 𝑅2
```python
rmse = 0
b0 = weights[0][0]
b1 = weights[1][0]

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2

rmse = np.sqrt(rmse/len(Y))
print(rmse)
```python
```python
ss_t = 0
ss_r = 0

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_r/ss_t)
print(r2)
```
