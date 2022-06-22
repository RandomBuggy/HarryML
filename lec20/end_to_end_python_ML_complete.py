"""
Author : RandomBuggy
Date : 23.02.2022
Purpose : ML Project - House Price Prediction
Packages : numpy pandas matplotlib scipy jupyter scikit-learn
Data : Housing Data Boston UCI ML Repository
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedSuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import SimpleImputer
from sklearn.preprocessing import StanderdScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

housing = pd.read_csv("housing.csv")
# print(housing.head())
# print(housing.info())
# print(housing["CHAS"].value_counts())
# print(housing.describe())

# %matplotlib inline
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

#train-test splitting
def split_train_test(data, test_ratio=0.2):
    # 42 is convention in ML
    np.random.seed(42)
    suffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    test_indices = suffled[0:test_set_size]
    train_indices = suffled[test_set_size:-1]

    return data.iloc[train_indices], data.iloc[test_indices]

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

# looking for co-relations
corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

# trying out attributes combinations
housing["TAXRM"] = housing["TAX"] / housing["RM"]
corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)
housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

#missing attributes
# to get rid of missing attributes, you have 3 options
    # 1. get rid of whole attributes
    # 2. set the value to 0 or mean or median
    # 3. missing values
a = housing.dropna(subset=["RM"]) #option 1
a.shape
housing.drop("RM", axis=1).shape #option 2
median = housing["RM"].median
housing["RM"].fillna(median) #option 3


if __name__ == "__main__":
    # train_set, test_set = split_train_test(housing, 0.2)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    print(f"Rows in train set : {len(train_indices)}\nRows in test set : {len(test_indices)}")
    split = StratifiedSuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["CHAS"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        housing = strat_train_set.copy()
    imputer = Imputer(strategy="median")
    imputer.fit(housing)
    # print(imputer.statistics_.shape)
    X = imputer.transform(housing)
    housing_tr = pd.DataFrame(X, columns=housing.columns)
    housing_tr.describe()
    # scikit-learn design

    # only 3 types of objects
    # 1. estimators - it estimators some parameter on a dataset. eg imputer
    # it has a fit method and transform method
    # fit method - fit the dataset and calculate internal parameters
    # 2. transformers - transform method takes input and return output based on the learning from fit(). 
    # it also has a convinience function called fit_transform() which fits then transform
    # 3. predictors - LinearRegression model is an example of predictor.
    # fit() and predict() are two common method
    # it will give score() method which will evaluate predictions



    # feature scaling
    # primarily two types of feature scaling method

    # 1. min-max scaling (normalization)
    # (value - min) / (max - min)
    # scikit-learn provides a library called MinMaxScaler
    # 2. standerdization
    # (value - mean) / std
    # sklearn provides a class called StanderdScaler





    # Creating a Pipeline
    my_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("std_scaler", StanderdScaler())])
    housing_tr_num = my_pipeline.fit_transform(housing)
    housing_tr_num.shape

    # selecting and training the model
    housing = strat_train_set.drop("MEDV", axis=1)
    housing_labels = strat_train_set["MEDV"].copy()
    model = RandomForestRegressor()
    # model = DecisionTreeRegressor()
    # model = LinearRegression()
    model.fit(housing, housing_labels)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]

    prepared_data = my_pipeline.transform(some_data)
    model.predict(prepared_data)

    housing_predictions = model.predict(housing_tr_num)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)

    # cross validation
    scores = cross_val_score(model, housing_tr_num, housing_labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    print_scores(rmse_scores)

    # joblib
    dump(model, "mymodel.joblib")

    #testing the model on test data
    X_test = strat_test_set.drop("MEDV", axis=1)
    Y_test = strat_test_set["MEDV"].copy()
    X_test_prepared = my_pipeline.transform(X_test)
    final_prediction = model.predict(X_test_prepared)
    final_mse = mean_squared_error(Y_test, final_prediction)
    final_rmse = np.sqrt(final_mse)

    # using model
    model = load("mymodel.joblib")
    feature = np.array([[4666, 5545.9765, -46644, bla bla bla]])
    model.predict(features)
