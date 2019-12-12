# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project, slash
from scripts.save_workspace import save
import statistics as st

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
base_classification = Base_Classification(hmp, extra_trees)

#===LOAD FEATURES===#

#Interate threshold to find de best value#

s = save()
person_list = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "m5", "m6", "m7", "f4", "m8", "m9", "f5", "m10", "m11"]
accuracy_threshould_temp_aux = pd.DataFrame(columns=["accuracy","discarted", "len_activity", "threshold"])
accuracy_mean = pd.DataFrame(columns=["accuracy","discarted", "len_activity", "threshold"])
project.log("====================HMP BEST THRESHOLD============================", file="hmp_log_best_threshold.log")
for t in np.arange(0.05, 1, 0.05):
    accuracy_threshould_temp_aux = pd.DataFrame(columns=["accuracy","discarted", "len_activity"])
    for p in person_list:
        relevant_features = s.load_var("hmp_relevant_features{}relevant_features_{}.pkl".format(slash, p))
        y = s.load_var("hmp_relevant_features{}y_{}.pkl".format(slash, p))
        y = pd.DataFrame(y, columns=[hmp.label_tag])
        X_train, X_test, y_train, y_test = train_test_split(relevant_features, y, test_size=0.2, random_state=42)
        data = {}
        data[p] = {}
        data[p]["training"] = {}
        data[p]["training"]["training_features"] = X_train
        data[p]["training"]["training_labels"] = y_train
        data[p]["test"] = {}
        data[p]["test"]["test_features"] = X_test
        data[p]["test"]["test_labels"] = y_test
        return_proba = base_classification.get_accuracy.simple_accuracy_mean_to_each_person_with_proba(data, hmp, extra_trees, t)
        return_proba = return_proba[list(return_proba.keys())[0]]
        accuracy_threshould_temp_aux = accuracy_threshould_temp_aux.append(return_proba, ignore_index=True)
        
    project.log("Accuracy: {}, Discarted: {}, Len_activity: {}, Threshold: {}".format(st.mean(accuracy_threshould_temp_aux["accuracy"]), st.mean(accuracy_threshould_temp_aux["discarted"]), st.mean(accuracy_threshould_temp_aux["len_activity"]), t), file="hmp_log_best_threshold.log")
    print("Accuracy: {}, Discarted: {}, Len_activity: {}, Threshold: {}".format(st.mean(accuracy_threshould_temp_aux["accuracy"]), st.mean(accuracy_threshould_temp_aux["discarted"]), st.mean(accuracy_threshould_temp_aux["len_activity"]), t))

project.log("================================================", file="hmp_log_best_threshold.log")