# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.arcma_model import ARCMA_Model
from classifiers.base_classification import Base_Classification
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project
from statistics import mean
#===== Machine Learn =====#
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn import svm #SVM
from sklearn.neural_network import MLPClassifier #multi-layer perceptro
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing.get_accuracy import Get_Accuracy
from scripts.save_workspace import save
import numpy as np
import statistics as st
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
threshold_balance_data = 100
#Select de best windows
t = time.time()
best_model = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
w_accuracies = pd.DataFrame(columns=["window", "accurary"])
p = 15 # pessoa com mais registros
project.log("=====================ARCMA_SELECT_BEST_WINDOWS=====================", file="arcma_log_best_window.log")
for w in range(10,110,10):
    print("Load data with window len = {}".format(w))
    data_list_people = arcma.load_training_data_from_list_people(w, [p], remove_outliers=0.05)
    dataframe_1 = data_list_people[p]["training"]
    dataframe_2 = pd.DataFrame()
    labels = []
    id = 1
    print("Load data in tsfresh input")
    for d in dataframe_1:
        if len(np.unique(d[arcma.label_tag])) < 2:
            d["id"] = pd.Series(np.full((1,d.shape[0]), id)[0], index=d.index)
            d["time"] = pd.Series(range((id-1)*d.shape[0], id*d.shape[0]), index=d.index)
            labels.append(d["activity"].iloc[0])
            dataframe_2 = dataframe_2.append(d, ignore_index=True)
            id = id + 1
    del dataframe_1  
    dataframe_3 = dataframe_2.drop(arcma.label_tag, 1)
    del dataframe_2
    y = np.array(labels)
    del labels
    y2 = pd.Series(y)
    del y
    
    print("Calculate relevant features .....")
    y2.index+= 1
    relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time')
    y2 = pd.DataFrame(y2, columns=[arcma.label_tag])
    
    balanced_data = balance_data.balance_data(relevant_features, y2, threshold_balance_data)
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
        del relevant_features, y2, x_train, x_test, y_train, y_test, test_valid_rows, train_valid_rows, accuracy, balanced_data
    break
project.log("===============================================================", file="arcma_log_best_window.log")
#base_classification = Base_Classification(arcma, ExtraTreesClassifier(n_estimators = 1000, random_state=0)) # Best algoritm
#best_window = base_classification.find_best_window(range(2, 100, 2), [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#dataframe = pd.DataFrame(best_window)

