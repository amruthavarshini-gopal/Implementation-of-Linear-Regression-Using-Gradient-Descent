# Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import necessary libraries for numerical operations, data handling, and preprocessing.

2.Load the startup dataset (50_Startups.csv) using pandas.

3.Extract feature matrix X and target vector y from the dataset.

4.Convert feature and target values to float and reshape if necessary.

5.Standardize X and y using StandardScaler.

6.Add a column of ones to X to account for the bias (intercept) term.

7.Initialize model parameters (theta) to zeros.

8.Perform gradient descent to update theta by computing predictions and adjusting for error.

9.Input a new data point, scale it, and add the intercept term.

10.Predict the output using learned theta, then inverse-transform it to get the final result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013  
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    ##coefficient of b
    x=np.c_[np.ones(len(x1)),x1]
    ##initialize theta with zero
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    ##perform gradient decent
    for _ in range(num_iters):
        ##calculate predictions
        predictions=(x).dot(theta).reshape(-1,1)
        ##calculate errors
        errors=(predictions - y).reshape(-1,1)
        ##update theta using gradient descent
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head())
##assume the last column as your target varible y
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
##learn model parameters
theta=linear_regression(x1_scaled,y1_scaled)
##predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
### Dataset values
![Screenshot 2025-04-28 141503](https://github.com/user-attachments/assets/0b5b19ca-cf54-4540-8692-d5a982e899f1)

### X Values
![Screenshot 2025-04-28 141641](https://github.com/user-attachments/assets/530210ac-bafe-4216-8907-a5aa361132f3)

![Screenshot 2025-04-28 141736](https://github.com/user-attachments/assets/d321af4a-9317-4a1e-8d31-0d769be389cc)

### Y Values
![Screenshot 2025-04-28 141918](https://github.com/user-attachments/assets/d8aa3da6-a63d-426a-a0a7-f1ec36941199)

![Screenshot 2025-04-28 141943](https://github.com/user-attachments/assets/3f7cc19a-4d9d-4902-babe-61bb37e8ecb5)

### X1_scaled Values
![Screenshot 2025-04-28 142153](https://github.com/user-attachments/assets/71152c63-89f7-4676-b392-8ce960b9d739)

![Screenshot 2025-04-28 142205](https://github.com/user-attachments/assets/db7b0eb1-812a-46cf-ab32-e6afc2771592)

### Y1_scaled Values
![Screenshot 2025-04-28 142217](https://github.com/user-attachments/assets/05c7931a-7e1a-4b6d-950e-0fa311fe58fb)

![Screenshot 2025-04-28 142230](https://github.com/user-attachments/assets/4e3f7c66-71fd-4de9-af0c-11989c72bbbb)

### Predicted Values
![Screenshot 2025-04-28 142241](https://github.com/user-attachments/assets/cfbd0a36-69a4-4e8a-b16a-211846953566)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
