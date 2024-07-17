import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('/content/Salary_Data.csv')
df
x = df['YearsExperience']
y = df['Salary']
plt.scatter(x,y)
plt.title('Salary vs YearsExperience')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
reg = LinearRegression()
reg.fit(x_train.values.reshape(-1,1),y_train)
y_train.shape
y_pred = reg.predict(x_test.values.reshape(-1,1))
r2_score(y_test,y_pred)
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)],color='red')
plt.title('Actual v/s Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show
