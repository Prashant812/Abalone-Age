# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:20:27 2022

@author: Prashant Kumar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures


print("\nSimple linear regression model\n")
df = pd.read_csv('abalone.csv')

#Splitting the data into test and train
[X_train, X_test] =train_test_split(df, test_size=0.30, random_state=42, shuffle = True)


#Saving the training and testing data in CSV files
X_train.to_csv("abalone-train.csv",index = False)
X_test.to_csv("abalone-test.csv",index = False)

#Finding the attribute which has the highest Pearson correlation coefficient with the target attribute Rings
corr = df.corr(method = 'pearson')
print(corr['Rings'])

#Function for simple linear (straight-line) regression model to predict rings
def linear_fit(data):
    regressor = LinearRegression()
    shell_weight = np.array(data['Shell weight'])
    shell_weight = shell_weight.reshape(-1,1)
    Rings = np.array(data['Rings'])
    Rings = Rings.reshape(-1,1)
    regressor.fit(shell_weight, Rings)#fitting training data
    y_pred = regressor.predict(shell_weight)
    return y_pred


#Best fit line between 'Shell weight' and 'Rings'
plt.title("Best Fit") #plotting the graph
plt.xlabel("Shell weight")
plt.ylabel("Rings")
plt.plot(X_train['Shell weight'], linear_fit(X_train))
plt.show()

#The prediction accuracy on the training and data using root mean squared error
error = sqrt(mean_squared_error(X_train['Rings'], linear_fit(X_train))) #computing the error in train data
error_test = sqrt(mean_squared_error(X_test['Rings'],linear_fit(X_test)))#computing the error in test data
print("RMSE Error for Linear Regression, Train data is : %.3f"%(error))
print("RMSE Error for Linear Regression, Test data is : %.3f"%(error_test))

#The scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data :-
plt.title("Scatter Plot") #plotting the graph
plt.xlabel("Rings(Actual)")
plt.ylabel("Rings(Predicted)")
plt.scatter(X_test['Rings'], linear_fit(X_test))
plt.show()



#Function for Multivariate linear regression model to predict rings
print("\nMultivariate linear regression model\n")
def multiple_fit(data):
    regressor = LinearRegression()
    input_var = np.array(data.iloc[:,:-1])
    target = np.array(data.iloc[:,-1])
    regressor.fit(input_var,target)
    pred = regressor.predict(input_var)
    return pred


#The prediction accuracy on the training and data using root mean squared error :-
error = sqrt(mean_squared_error(X_train['Rings'], multiple_fit(X_train))) #computing the error in train data
error_test = sqrt(mean_squared_error(X_test['Rings'],multiple_fit(X_test)))#computing the error in test data
print("RMSE Error for Multiple Regression, Train data is : %.3f"%(error))
print("RMSE Error for Multiple Regression, Test data is : %.3f"%(error_test))

#The scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data :-
plt.title("Scatter Plot for multivariate regression") #plotting the graph
plt.xlabel("Rings(Actual)")
plt.ylabel("Rings(Predicted)")
plt.scatter(X_test['Rings'], multiple_fit(X_test))
plt.show()
    

#Simple nonlinear regression model using polynomial curve fitting to predict Rings :-
print("\nSimple nonlinear regression model using polynomial curve fitting\n")
def poly_pred(data):#Function for prediction of data
    l = []
    for i in [2,3,4,5]:
        polynomial_features = PolynomialFeatures(degree = i)
        shell_weight = np.array(data['Shell weight'])
        shell_weight = shell_weight.reshape(-1,1)
        Rings = np.array(data['Rings'])
        Rings = Rings.reshape(-1,1)
        x_poly = polynomial_features.fit_transform(shell_weight)
        regressor = LinearRegression()
        regressor.fit(x_poly, Rings)
        y_pred = regressor.predict(x_poly)
        error = sqrt(mean_squared_error(data['Rings'], y_pred))
        l.append(error)
    ls = [round(x,3) for x in l]
    return [ls,y_pred]


#Prediction accuracy on the training and test data for the different values of degree of the polynomial (p = 2, 3, 4, 5) using root mean squared error (RMSE)
p = [2,3,4,5]
print("\nRMSE for Training data :-")
for i in range(len(p)):
    print("RMSE Error for p = %d is %f"%(p[i],poly_pred(X_train)[0][i]))
print("\nRMSE for Test data :-")
for i in range(len(p)):
    print("RMSE Error for p = %d is %f"%(p[i],poly_pred(X_test)[0][i]))


##function for plotting RMSE vs degree of polynomial
def plot2(data,x,y): 
    plt.title('Bar graph of RMSE vs degree of polynomial for %s'%(data))
    plt.xlabel('Degree of polynomial')
    plt.ylabel('RMSE')
    plt.bar(x, y, color = 'orange', width = .50)
    plt.show()

#Plotting the RMSE vs degree of polynomial graph :-
plot2('Traning Data',[2,3,4,5],poly_pred(X_train)[0])
plot2('Test Data',[2,3,4,5],poly_pred(X_test)[0])

#plotting scatter plot for best fit polynomial
plt.title('Best Fit Polynomial') 
plt.xlabel('Shell weight')
plt.ylabel("Predicted Rings")
plt.scatter(X_train['Shell weight'],poly_pred(X_train)[1])
plt.show()

#the scatter plot of the actual number of Rings (x-axis) vs the predicted number of Rings (y-axis) on the test data for the best degree of the polynomial (p) :-
plt.title('Predicted vs Real')
plt.xlabel('Actual Rings')
plt.ylabel("Predicted Rings")
plt.scatter(X_test['Rings'],poly_pred(X_test)[1])
plt.show()

#Multivariate nonlinear regression model using polynomial regression to predict Rings :-
print("\nMultivariate nonlinear regression model using polynomial curve fitting\n")
def multi_poly_pred(data):#Function for prediction of data
    l = []
    for i in [2,3,4,5]:
        polynomial_features = PolynomialFeatures(degree = i)
        input_var = np.array(data.iloc[:,:-1])
        x_poly = polynomial_features.fit_transform(input_var)
        Rings = np.array(data['Rings'])
        regressor = LinearRegression()
        regressor.fit(x_poly, Rings)
        y_pred = regressor.predict(x_poly)
        error = sqrt(mean_squared_error(data['Rings'], y_pred))
        l.append(error)
    ls = [round(x,3) for x in l]
    return [ls,y_pred]


#The prediction accuracy on the training and test data for the different values of degree of the polynomial (p = 2, 3, 4, 5) using root mean squared error (RMSE) :-
p = [2,3,4,5]
print("\nRMSE for Training data :-")
for i in range(len(p)):
    print("RMSE Error for p = %d is %f"%(p[i],multi_poly_pred(X_train)[0][i]))
print("\nRMSE for Test data :-")
for i in range(len(p)):
    print("RMSE Error for p = %d is %f"%(p[i],multi_poly_pred(X_test)[0][i]))

#Plotting the RMSE vs degree of polynomial graph :-
plot2('Traning Data',[2,3,4,5],multi_poly_pred(X_train)[0])
plot2('Test Data',[2,3,4,5],multi_poly_pred(X_test)[0])

#Plotting the best fit curve using the best fit model on the training data :-
plt.title('Predicted vs Real')
plt.xlabel('Actual Rings')
plt.ylabel("Predicted Rings")
plt.scatter(X_test['Rings'],multi_poly_pred(X_test)[1])
plt.show()
