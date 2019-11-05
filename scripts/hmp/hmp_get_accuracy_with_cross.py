# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
import pandas as pd
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project
from scripts.save_workspace import save
from pre_processing.get_accuracy import Get_Accuracy
from sklearn.model_selection import StratifiedKFold
import numpy as np

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 10000, random_state=0)
get_accuracy = Get_Accuracy()

#===LOAD FEATURES===#

#Interate threshold to find de best value#

s = save()
relevant_features = s.load_var("hmp_relevant_features\\relevant_features_m2.pkl")
y = s.load_var("hmp_relevant_features\\y_m2.pkl")
y = pd.DataFrame(y, columns=[hmp.label_tag])
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

accuracy = {}
accuracies = []
for train_index, test_index in skf.split(relevant_features, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = relevant_features.loc[train_index], relevant_features.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    test_valid_rows = np.isfinite(x_test["y__quantile__q_0.8"])
    train_valid_rows = np.isfinite(x_train["y__quantile__q_0.8"])
    
    x_test = x_test[test_valid_rows]
    y_test = y_test[test_valid_rows]
    x_train = x_train[train_valid_rows]
    y_train = y_train[train_valid_rows]
    
    extra_trees = ExtraTreesClassifier(n_estimators = 10000, random_state=0)
    threshold = 0.55
    accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, extra_trees, threshold)
    accuracies.append(accuracy)
    
accuracy_mean = np.mean([a["accuracy"] for a in accuracies])
discarted_mean = np.mean([a["discarted"] for a in accuracies])

print("Accuracy Mean: {}".format(accuracy_mean))
print("Discarted Mean: {}".format(discarted_mean))    