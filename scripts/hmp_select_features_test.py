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
import itertools  
from sklearn.metrics import accuracy_score
from utils.project import Project
import time

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, max_depth=1000, random_state=0) #Good performer
base_classification = Base_Classification(hmp, extra_trees)
_, _, _ = base_classification.predict_outliers_for_list_people_with_proba(50, ["f1", "m1", "m2"], "eat_soup" ,0.55, remove_outliers=0.05)

#===Extract TsFresh Features===#
dataframe_1 = hmp.data_with_window["f1"]["training"]
dataframe_2 = pd.DataFrame()
labels = []
id = 1
for d in dataframe_1:
    if len(np.unique(d[hmp.label_tag])) < 2:
        d["id"] = pd.Series(np.full((1,d.shape[0]), id)[0], index=d.index)
        d["time"] = pd.Series(range((id-1)*d.shape[0], id*d.shape[0]), index=d.index)
        labels.append(d["activity"].iloc[0])
        dataframe_2 = dataframe_2.append(d, ignore_index=True)
        id = id + 1
del dataframe_1  
dataframe_3 = dataframe_2.drop(hmp.label_tag, 1)
del dataframe_2
y = np.array(labels)
del labels
y2 = pd.Series(y)
del y

y2.index+= 1
time.sleep(10)
#----------------------------------

# 1º - VERIFICAR ACURÁCIA UTILIZANDO TODO ts_features
relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time')
X_train, X_test, y_train, y_test = train_test_split(relevant_features, y2, test_size=0.2, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0)
extra_trees.fit(X_train, y_train)
start_time = time.time()
pred = extra_trees.predict(X_test)
end_time = time.time()
accuracy = accuracy_score(y_test, pred)
project.log("Accuracy to all ts_features ({}): {} - Time: {} seconds.".format(len(relevant_features.columns), accuracy, (end_time - start_time)/len(y_test)))
time.sleep(10)
del X_train, X_test, y_train, y_test, extra_trees, start_time, pred, accuracy
time.sleep(10)

# 2º - ACURÁCIA COM OS 10% MAIS RELEVANTES ts_features
ts_extratree_features_importance = pd.DataFrame(extra_trees.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
len_features = len(ts_extratree_features_importance)
best_features = ts_extratree_features_importance.index[0:int((len_features/10)-1)]
ts_final_features = relevant_features[best_features]

extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(ts_final_features, y2, test_size=0.2, random_state=42)
extra_trees.fit(X_train, y_train)
start_time = time.time()
pred = extra_trees.predict(X_test)
end_time = time.time()
accuracy = accuracy_score(y_test, pred)
project.log("Accuracy to all 10% ts_features ({}): {} - Time: {} seconds.".format(len(ts_final_features.columns), accuracy, (end_time - start_time)/len(y_test)))
time.sleep(10)
del X_train, X_test, y_train, y_test, extra_trees, start_time, pred, accuracy, ts_extratree_features_importance, ts_final_features
time.sleep(10)

# 3º - ACURÁCIA COM initial_features
initial_features = processing.calculating_features_raw(hmp.data_with_window["f1"]["training"], hmp.label_tag, hmp.features[0], hmp.features[1], hmp.features[2])
initial_features[0].index+= 1

X_train, X_test, y_train, y_test = train_test_split(initial_features[0], y2, test_size=0.2, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0)
extra_trees.fit(X_train, y_train)
start_time = time.time()
pred = extra_trees.predict(X_test)
end_time = time.time()
accuracy = accuracy_score(y_test, pred)
project.log("Accuracy to initial_features ({}): {} - Time: {} seconds.".format(len(initial_features[0].columns), accuracy, (end_time - start_time)/len(y_test)))
time.sleep(10)
del X_train, X_test, y_train, y_test, extra_trees, start_time, pred, accuracy
time.sleep(10)

# 4º ACURÁCIA COM ts_features + initial_features
# Add initial features in to ts_final_features
for i in initial_features[0].keys():
    relevant_features[i] = pd.Series(initial_features[0][i], index=relevant_features.index)

X_train, X_test, y_train, y_test = train_test_split(relevant_features, y2, test_size=0.2, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0)
extra_trees.fit(X_train, y_train)
start_time = time.time()
pred = extra_trees.predict(X_test)
end_time = time.time()
accuracy = accuracy_score(y_test, pred)
project.log("Accuracy to ts_features + initial_features ({}): {} - Time: {} seconds.".format(len(relevant_features.columns), accuracy, (end_time - start_time)/len(y_test)))
time.sleep(10)
del X_train, X_test, y_train, y_test, extra_trees, start_time, pred, accuracy
time.sleep(10)
# 5º ACURÁCIA COM 21 MAIS RELEVANTE ts_features

# 6º ACURÁCIA COM 21 MAIS RELEVANTE ts_features + initial_features


#search for the best combination
'''subset = []
for i in range(2, 401):
    columns = list(itertools.combinations(ts_final_features.keys(), i))
    for c in columns:
        c = list(c)
        subset = ts_final_features[c]
        break
'''
from scripts.save_workspace import save
s = save()
s.save_var(relevant_features, "Relevant_features.pkl")
s.save_var(y2, "y.pkl")

