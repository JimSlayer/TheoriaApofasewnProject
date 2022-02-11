import pandas as pd 
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, r2_score

# Read data from file 'Dataset_1.csv' 
data = pd.read_csv("C:/Users/Ceid/Desktop/pytho_project/Dataset_1.csv") 
#print(data)

# Επιλογή χαρακτηριστικών για το train
X = data.drop(['trestbps','chol','thalach','target'],axis = 1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# print train dataset
print('Training DataSet: ')
print(X_train)
print('\n')

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X_train,y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# print predictions and results
print('Predictions DataSet: ')
print(y_pred)
print('\n')

print('Results DataSet: ')
print(y_test)
print('\n')

# The mean squared error (Ελάχιστα Τετράγωνα)
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.plot(y_pred, y_test, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
