# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
import pandas as pd
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project, slash
from scripts.save_workspace import save
from pre_processing.get_accuracy import Get_Accuracy
from sklearn.model_selection import StratifiedKFold
from pre_processing.balance_data import BalanceData
import numpy as np

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 40

#===LOAD FEATURES===#

#Interate threshold to find de best value#
persons = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "m5", "m6", "m7", "f4", "m8", "m9", "f5", "m10", "m11"]
accuracy_by_person = pd.DataFrame()
threshold = 0.65
project.log("=========== HMP StratifiedKFold Accuracy, Thresold = {}===========".format(threshold), file="hmp_log_final_accuracy.log")
for p in persons:
    s = save()
    relevant_features = s.load_var("hmp_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("hmp_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[hmp.label_tag])
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    
    accuracy = {}
    accuracies = []
    balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
    if isinstance(balanced_data, tuple):
        for train_index, test_index in skf.split(relevant_features, y):
            x_train, x_test = balanced_data[0].loc[train_index], balanced_data[0].loc[test_index]
            y_train, y_test = balanced_data[1].loc[train_index], balanced_data[1].loc[test_index]
            
            test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
            train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
            
            x_test = x_test[test_valid_rows]
            y_test = y_test[test_valid_rows]
            x_train = x_train[train_valid_rows]
            y_train = y_train[train_valid_rows]
            
            extra_trees = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
            accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, extra_trees, threshold)
            accuracies.append(accuracy)
    else:
        continue
    accuracy_mean = np.mean([a["accuracy"] for a in accuracies])
    discarted_mean = np.mean([a["discarted"] for a in accuracies])
    
    accuracy_by_person = accuracy_by_person.append(pd.DataFrame([{"person": p, "accuracy":accuracy_mean, "discarted":discarted_mean}]))
    
    project.log("Person {} - Accuracy Mean: {} | Discarted Mean: {}".format(p, accuracy_mean,discarted_mean), file="hmp_log_final_accuracy.log")