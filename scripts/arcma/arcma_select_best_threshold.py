# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.arcma_model import ARCMA_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project, slash
from scripts.save_workspace import save
import statistics as st
from pre_processing.balance_data import BalanceData

#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
base_classification = Base_Classification(arcma, extra_trees)
balance_data = BalanceData()
threshold_balance_data = 40

#Interate threshold to find de best value#

s = save()
person_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
accuracy_threshould_temp_aux = pd.DataFrame(columns=["accuracy","discarted", "len_activity", "threshold"])
accuracy_mean = pd.DataFrame(columns=["accuracy","discarted", "len_activity", "threshold"])
project.log("====================ARCMA BEST THRESHOLD============================", file="arcma_log_best_threshold.log")
for t in np.arange(0.05, 1, 0.05):
    accuracy_threshould_temp_aux = pd.DataFrame(columns=["accuracy","discarted", "len_activity"])
    for p in person_list:
        relevant_features = s.load_var("arcma_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
        y = s.load_var("arcma_relevant_features_best_window{}y_{}.pkl".format(slash, p))
        y = pd.DataFrame(y, columns=[arcma.label_tag])
        
        balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
        if isinstance(balanced_data, tuple):
            x_train, x_test, y_train, y_test = train_test_split(balanced_data[0], balanced_data[1], test_size=0.2, random_state=42)
            data = {}
            data[p] = {}
            data[p]["training"] = {}
            data[p]["training"]["training_features"] = x_train
            data[p]["training"]["training_labels"] = y_train
            data[p]["test"] = {}
            data[p]["test"]["test_features"] = x_test
            data[p]["test"]["test_labels"] = y_test
            return_proba = base_classification.get_accuracy.simple_accuracy_mean_to_each_person_with_proba(data, arcma, extra_trees, t)
            return_proba = return_proba[list(return_proba.keys())[0]]
            accuracy_threshould_temp_aux = accuracy_threshould_temp_aux.append(return_proba, ignore_index=True)
        else:
            print("Threshold {} - Person {} - not enough records")
        #break   
    project.log("accuracy: {}, discarted_register: {}, discarted_activity: {}, threshold: {}".format(st.mean(accuracy_threshould_temp_aux["accuracy"]), st.mean(accuracy_threshould_temp_aux["discarted"]), st.mean(accuracy_threshould_temp_aux["discarted_activity"]), t), file="arcma_log_best_threshold.log")
    print("accuracy: {}, discarted_register: {}, discarted_activity: {}, threshold: {}".format(st.mean(accuracy_threshould_temp_aux["accuracy"]), st.mean(accuracy_threshould_temp_aux["discarted"]), st.mean(accuracy_threshould_temp_aux["len_activity"]), t))
    #break
project.log("================================================", file="arcma_log_best_threshold.log")