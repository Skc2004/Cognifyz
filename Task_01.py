# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Dataset.csv')


print(df.head())

df.fillna(df.mean(), inplace=True)  


X = df.drop('Aggregate rating', axis=1)  
y = df['Aggregate rating'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


if isinstance(model, LinearRegression):
    print("\nCoefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(feature, ':', coef)


