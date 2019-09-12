# -*- coding: utf-8 -*-
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np

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
id = 1
for d in dataframe_1:
    d["id"] = pd.Series(np.full((1,d.shape[0]), id)[0], index=d.index)
    d["time"] = pd.Series(range(id-1, id*d.shape[0]), index=d.index)
    dataframe_2 = dataframe_2.append(d, ignore_index=True)
    id = id + 1
    if id > 2:
        break



