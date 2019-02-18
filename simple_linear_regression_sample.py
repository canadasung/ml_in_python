# Multiple Linear Regression
# 1 to 1 analysis, there is only 1 dependent variable (y) and 1 independent variable (x)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
dataset.info()
# For usage of iloc
# See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# Always make sure X is a matrix and Y is a vector
X = dataset.iloc[:, :-1].values # select all columns except the last one
y = dataset.iloc[:, 1].values # select second (index 1) column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling isn't needed here because simple linear regresssion model
# will take care of it for us.


# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Visualising the Training Set Result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set Result
plt.scatter(X_test, y_test, color = 'red')

'''we don't need to change the regressor.predict to X_test
because the regression line will be the same'''
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Run Statistical Results
import statsmodels.api as sm # Display Ordinary Least Square Regression Results

# Due the nature of OLS, we need to create a constant 1 as X0 variable to satisfy the regression formula
X_with_constant = sm.add_constant(X)

# Run OLS Result
X_opt = X_with_constant
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()