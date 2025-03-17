# import all the lib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# read the dataset using pandas
data = pd.read_csv(r'D:\data set\Salary_Data.csv')
# This displays the top 5 rows of the data
     #data.head()
# Provides some information regarding the columns in the data
    #data.info()
# this describes the basic stat behind the dataset used 
    #data.describe()

# These Plots help to explain the values and how they are scattered
X = data[["YearsExperience"]].values  
y = data["Salary"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="blue", alpha=0.6, edgecolors="k", label="Actual Salaries")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Actual Salaries (Test Set)")
plt.legend()
plt.grid(True)
plt.show()

# Cooking the data
X = data['YearsExperience']
X.head()

# Cooking the data
y = data['Salary']
y.head()


# Split the data for train and test (70% for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# Create new axis for x column
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]


# Importing Linear Regression model from scikit learn
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)


# Predicting the Salary for the Test values
y_pred = model.predict(X_test)
print("Predicted Salaries:", y_pred)


# Plotting the actual and predicted values
c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()



# plotting the error
c = [i for i in range(1,len(y_test)+1,1)]
plt.plot(c,y_test-y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()



# Importing r2_score and mean_squared_error for the evaluation of the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R-squared (R²) Score:", r2)
print("Mean Squared Error (MSE):", mse)




# calculate Mean square error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)



# Calculate R square vale
r2 = r2_score(y_test, y_pred)
print("R-squared (R²) Score:", r2)




# Just plot actual and predicted values for more insights
plt.figure(figsize=(12,6))
plt.scatter(y_test,y_pred,color='r',linestyle='-')
plt.show()
