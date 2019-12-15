# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.arcma_model import ARCMA_Model
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing.get_accuracy import Get_Accuracy
import numpy as np
from tsfresh import extract_relevant_features
import time
from pre_processing.balance_data import BalanceData

#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 40
#Select de best windows
t = time.time()
best_model = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
w_accuracies = pd.DataFrame(columns=["window", "accurary"])
p = 15 # pessoa com mais registros
project.log("=====================ARCMA_SELECT_BEST_WINDOWS=====================", file="arcma_log_best_window.log")
for w in range(20,110,10):
    
    print("Load data with window len = {}".format(w))
    data = arcma.load_training_data_by_people(p)
    print("Slicing Window....")
    data_tsfresh, y = arcma.slice_by_window_tsfresh(data, w)
    y.index += 1
    del data_tsfresh["activity"]
    
    classes_counts = y.value_counts()
    if len(classes_counts) > 1:
        relevant_features = extract_relevant_features(data_tsfresh, y, column_id='id', column_sort='time')
        y = pd.DataFrame(y, columns=[arcma.label_tag])
    
        balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
        if isinstance(balanced_data, tuple):
            x_train, x_test, y_train, y_test = train_test_split(balanced_data[0], balanced_data[1], test_size=0.2, random_state=42)
            
            test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
            train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
                
            x_test = x_test[test_valid_rows]
            y_test = y_test[test_valid_rows]
            x_train = x_train[train_valid_rows]
            y_train = y_train[train_valid_rows]
        
            accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, best_model, 0)["accuracy"]
            project.log("Window = {} | Accouracy = {}".format(w, accuracy), file="arcma_log_best_window.log")
            print("Finish to calc windows = {}".format(w))
            del x_train, x_test, y_train, y_test, test_valid_rows, train_valid_rows, accuracy
        del relevant_features, y
    del data, data_tsfresh

project.log("===============================================================", file="arcma_log_best_window.log")

