# Multiple Linear Regression
# Many to 1 analysis, there is only 1 dependent variable (y) and multiple independent variables (X)
# 6 Major Steps: 
#### 1. Import libraries
#### 2. Import dataset
#### 3. Encoding categorical variables if there is
#### 4. Split to training and test set
#### 5. Train model
#### 6. Predict using test set

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset.info()
# For usage of iloc
# See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# Always make sure X is a matrix and Y is a vector
X = dataset.iloc[:, :-1].to_numpy() # select all columns except the last one as x
y = dataset.iloc[:, 4].to_numpy() # select fifth (index 4) column as y

# Or manually select X and y using DataFrame
#X = dataset[['c1','c2','c3',...,'cn']]
#y = dataset['target']


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

# Encoding categorical data Version 2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling isn't needed here because simple linear regresssion model will take care of it for us.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Simple Linear Regression to the Training Set
# The Multiple Linear Regression Class will take care of dummy variable trap (the additional 1 dummy) for us.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results if using iloc + to_numpy()
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Predicting the Test Set Results Version 2 if using DataFrame
y_pred = regressor.predict(X_test)
plt.scatter(y_test, y_pred)
sns.histplot(y_test - y_pred)

# Making a single prediction
print(regressor.predict([[attr1, attr2, attr3,..., attrn]]))
"""
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. 
That's because the "predict" method always expects a 2D array as the format of its inputs. 
And putting our values into a double pair of square brackets makes the input exactly a 2D array. Simply put:
1,0,0,160000,130000,300000 → scalars 
[1,0,0,160000,130000,300000] → 1D array 
[[1,0,0,160000,130000,300000]] → 2D array 

Important note 2: Notice also that the "California" state was not input as a string in the last column but as "1, 0, 0" in the first three columns. 
That's because of course the predict method expects the one-hot-encoded values of the state, 
and as we see in the second row of the matrix of features X, "California" was encoded as "1, 0, 0". 
And be careful to include these values in the first three columns, not the last three ones, because the dummy variables are always created in the first columns.
"""
# Getting the linear regression equation
# y = b0 + b1 x1 + b2 x2 + ... + bn xn
# Target = intercept + coef1 * Attr1 + coef2 * Attr2 + coef3 * Attr3 + ... + coefn * Attrn
print(lm.intercept_)
print(lm.coef_)



########## EXTRA ##########
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
  
 
