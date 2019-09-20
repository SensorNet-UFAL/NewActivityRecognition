# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0) #Good performer
base_classification = Base_Classification(hmp, extra_trees)
return_dataframe, return_accuracy, data_from_each_person = base_classification.predict_outliers_for_list_people_with_proba(50, ["f1", "m1", "m2"], "eat_soup" ,0.55, remove_outliers=0.05)

#===Extract TsFresh Features===#
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
y = np.array(labels)
y2 = pd.Series(y)
y2.index+= 1
relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time')
X_train, X_test, y_train, y_test = train_test_split(relevant_features, y, test_size=0.2, random_state=42)
extra_trees.fit(X_train, y_train)
ts_extratree_features_importance = pd.DataFrame(extra_trees.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

#==Selecting 10% from the best TsFresh Features===#
len_features = len(ts_extratree_features_importance)
best_features = ts_extratree_features_importance.index[0:int((len_features/10)-1)]
ts_final_features = relevant_features[best_features]

#===Get initial features===#
initial_featues = processing.calculating_features_raw(dataframe_1, hmp.label_tag, hmp.features[0], hmp.features[1], hmp.features[2])
