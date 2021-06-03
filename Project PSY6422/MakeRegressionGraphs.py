#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:14:58 2019

Script to import, group and plot life satisfaction and alcohol consumption data, to observe the relationship

Code Project for PSY6422
@author: Stelz
"""
#---> Import Libraries used
import operator #number function, will be used for fixing the x variable for regression
import os #file functions
import pandas as pd #dataframes 
from sklearn.linear_model import LinearRegression#for regressions
from sklearn.preprocessing import PolynomialFeatures#for regression polynomial
import numpy as np #number functions
import matplotlib.pyplot as plt #plotting function

os.path.join('Project PSY6422','Data') #setting the path and work directory

#--> Import the raw data
df1=pd.read_csv('alcohol_consumption_data.csv').replace(' ', np.NaN)#-->replacing all empty cells with NaN
df2=pd.read_csv('life_satisfaction_data.csv').replace(' ', np.NaN)


df1 #check our data--> seems there are incomplete rows we don't want
df2 #check our data--> seems there are incomplete rows we don't want

#Start of data Munging
#dropping the rows that contain NaN, which we don't need
df1=df1.dropna(axis=0)
df2=df2.dropna(axis=0)

#creating new dataframes for individual questionnaires
dfaudit=df1[df1.columns[3:13]].astype(int)
dfrelationships=df2[df2.columns[3:12]].astype(int)
dfaccomplishment=df2[df2.columns[12:18]].astype(int)
#--> we also converted the numbers to integers so we can use them

#we want to get the mean questionnaire scores for each participant
dfaudit['mean_alcohol_consumption']=dfaudit.mean(axis=1)
dfrelationships['mean_rel']=dfrelationships.mean(axis=1)
dfaccomplishment['mean_acc']=dfaccomplishment.mean(axis=1)

#now we have 2 dataframes for life satisfaction and 1 for alcohol consumption.
#We want to combine the Life Satisfaction scores into 1 dataframe so we can calculate the total life satisfaction score
dflifesat=pd.concat([dfaccomplishment['mean_acc'],dfrelationships['mean_rel']],axis=1)#---> found this function by googling how to combine different columns from different data frames in pandas, a more basic concat function was given, and by playing around i managed to make it as tidy as possible
dflifesat['total_lifesat']=dflifesat.sum(axis=1)#calculating the total life satisfaction score

#now its time to prepare our data for plotting, by creating new variables
x=dfaudit['mean_alcohol_consumption'].values.reshape(-1, 1) #reshaping required for regression
y=dflifesat['total_lifesat'].values.reshape(-1, 1)
y1=dflifesat['mean_acc']
y2=dflifesat['mean_rel']
       
#plotting with linear regression
plt.plot(x,y, linestyle='none', color='blue', marker='p', markersize=10, markeredgecolor='yellow', markeredgewidth=2)
plt.title('Relationship of Life Satisfaction and alcohol consumption')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Life Satisfaction')
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
y_pred = linear_regressor.predict(x)  # make predictions
plt.plot(x, y_pred, color='orange') #regression line
plt.savefig('Lifesat-AlcoholConsumption_linearRegress.png')
plt.show()

#so after figuring out how to do a linear regression with the data, I then found out it was better to do a polynomial regression... so i did

#preparing variables for the polymodial regression
polynomial_features=PolynomialFeatures(degree=2) #creating an object for the class
x_poly=polynomial_features.fit_transform(x)
model=LinearRegression() #changing the object name for the function
model.fit(x_poly,y)
y_poly_pred=model.predict(x_poly)

#plotting the regression
plt.plot(x,y, linestyle='none', color='blue', marker='p', markersize=10, markeredgecolor='yellow', markeredgewidth=2)
plt.title('Relationship of Life Satisfaction and alcohol consumption')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Life Satisfaction')
#sorting the x values, creating the curve
sort_axis=operator.itemgetter(0)
sorted_zip=sorted(zip(x,y_poly_pred),key=sort_axis)
x,y_poly_pred=zip(*sorted_zip)
plt.plot(x,y_poly_pred, color='orange')
plt.savefig('Polynomial-regression_Lifesat-AlcoholConsumption.png')
plt.show()

#so even with the polynomial regression there isn't really a big effect on life satisfaction and alcohol consumption.. . 

