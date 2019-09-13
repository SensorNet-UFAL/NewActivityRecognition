# -*- coding: utf-8 -*-
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
from classifiers.base_classification import Base_Classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### First test, delet after real test work ###
'''
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()
print(timeseries.head())
timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True, figsize=(10,10))
plt.show()

extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
impute(extracted_features)
features_filtered = select_features(extracted_features, y)
'''

### Real Work ###

# First load hmp data in extratrees_classification.py
dataframe_1 = hmp.data_with_window["f1"]["training"]
dataframe_2 = pd.DataFrame()
labels = []
id = 1
for d in dataframe_1:
    d["id"] = pd.Series(np.full((1,d.shape[0]), id)[0], index=d.index)
    d["time"] = pd.Series(range((id-1)*d.shape[0], id*d.shape[0]), index=d.index)
    labels.append(d["activity"].iloc[0])
    dataframe_2 = dataframe_2.append(d, ignore_index=True)
    id = id + 1


dataframe_3 = dataframe_2.drop("activity", 1)
extracted_features = extract_features(dataframe_3, column_id="id", column_sort="time")
impute(extracted_features)
y = np.array(labels)
y2 = pd.Series(y)
features_filtered = select_features(extracted_features, y)

X_train, X_test, y_train, y_test = train_test_split(features_filtered, y, test_size=0.33, random_state=42)

extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0)
extra_trees.fit(X_train, y_train)
pred = extra_trees.predict(X_test)
accuracy_score(y_test, pred)
relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time') # error: ValueError: The following ids are in the time series container but are missing in y: {3480}
