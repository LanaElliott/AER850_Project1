# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:50:16 2023

@author: Owner
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data\\Project 1 Data.csv")

X = df['X']
Y = df['Y']
Z = df['Z']
step = df['Step']



#Prevent data breakage by dropping all n/a values
df = df.dropna()

my_scaler = StandardScaler()
my_scaler.fit(df)

#Step 2 Plotting the 3D graph and perfomre statistical analysis
print("")
print("Step 2, Data Visualization")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#To give color variation based on step size
ax.scatter(X, Y, Z, c=step, cmap='viridis')
#Labeling the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#Show the plot
plt.show()
#Analyzing and printing summary statistics
print(df.describe())


