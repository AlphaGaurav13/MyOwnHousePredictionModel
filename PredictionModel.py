#loading the model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

house_price_dataset  = fetch_california_housing()


house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)

house_price_dataframe['price'] = house_price_dataset.target

correlation = house_price_dataframe.corr()

X = house_price_dataframe.drop(['price'], axis=1)

Y = house_price_dataframe['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
model = XGBRegressor()

#training the model with X-train
model.fit(X_train, Y_train)

#evaluation
training_data_prediction = model.predict(X_train)
test_data_prediction = model.predict(X_test)

# r square error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

#Mean Absolute error

score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R square error : ", score_1)
print("Mean Absolute error : ", score_2)

#now see it on Graph
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()