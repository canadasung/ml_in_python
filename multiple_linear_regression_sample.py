# Multiple Linear Regression
# Many to 1 analysis, there is only 1 dependent variable (y) and multiple independent variables (x)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset.info()
# For usage of iloc
# See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# Always make sure X is a matrix and Y is a vector
X = dataset.iloc[:, :-1].values # select all columns except the last one as x
y = dataset.iloc[:, 4].values # select fifth (index 4) column as y

# Encoding categorical data to dummy variable (linear regression only accept numbers)
# Encoding the Independent Variable
# E.g.// each city is encoded a value
# Encoding must be done before splitting dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
""" In python, the library would take care of the dummy variable trap
for us. We remove a column here is just to demonstrate how to do it, which
might be needed in other programs. So, this step could be ignored when we work
in python.
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling isn't needed here because simple linear regresssion model
# will take care of it for us.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm # Display Ordinary Least Square Regression Results

# Due the nature of OLS, we need to create a constant 1 as X0 variable to satisfy the regression formula
# Original form is as following which add 1s to the last column
# X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)
# Since we wants 1s to be the first column, we reverse the function
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

############# More Efficient Method #############
# Run Statistical Results
import statsmodels.api as sm # Display Ordinary Least Square Regression Results
# Due the nature of OLS, we need to create a constant 1 as X0 variable to satisfy the regression formula
X_with_constant = sm.add_constant(X)
#################################################

# Run Backword Elimination by removing the highest P-value variable
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
  
 