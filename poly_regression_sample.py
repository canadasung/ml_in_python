# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset.info()
# For usage of iloc
# See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# Always make sure X is a matrix and Y is a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Avoiding the Dummy Variable Trap
# X = X[:, 1:]
""" In python, the library would take care of the dummy variable trap
for us. We remove a column here is just to demonstrate how to do it, which
might be needed in other programs. So, this step could be ignored when we work
in python.
"""

# Splitting the dataset into the Training set and Test set
# Remove here because the dataset is too small for this data case
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling isn't needed here because simple linear regresssion model
# will take care of it for us.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Simple Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Change degree accordingly
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression Results
plt.scatter(X, y, color = 'red') # Real observation
plt.plot(X, lin_reg.predict(X), color = 'blue') # Linear Prediction
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red') # Real observation
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') # Polynomial Prediction
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression results
y_lin_pred = lin_reg.predict(6.5)
y_lin_pred

# Predicting a new result with Polynomial Regression results
y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform(6.5))
y_poly_pred
 
 