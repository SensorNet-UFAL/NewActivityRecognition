# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.arcma_model import ARCMA_Model
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
#===== Machine Learn =====#
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn import svm #SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing.get_accuracy import Get_Accuracy
from scripts.save_workspace import save
import numpy as np
from pre_processing.balance_data import BalanceData
import statistics as st
import time


#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
classifiers = {"Extratrees": ExtraTreesClassifier(n_estimators = 1000), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(n_estimators = 1000), "Decision Tree":tree.DecisionTreeClassifier(), "SVM":svm.SVC(probability=True)}
persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 40
#Select the best classifier
accuracy_mean = pd.DataFrame(columns=["Classifier", "Accuracy"])
project.log("=====================ARCMA_SELECT_BEST_ALGORITHM=====================", file="arcma_best_algorithm.log")
for c in classifiers:
    print(c)
    person_accuracies = []
    times_to_predict = []
    for p in persons:
        s = save()
        try:
            relevant_features = s.load_var("arcma_relevant_features_fix_window{}relevant_features_{}.pkl".format(slash, p))
            y = s.load_var("arcma_relevant_features_fix_window{}y_{}.pkl".format(slash, p))
            y = pd.DataFrame(y, columns=[arcma.label_tag])
        except:
            print("file from person {} not found!".format(p))
            continue
        
        
        balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
        if isinstance(balanced_data, tuple):
        
            x_train, x_test, y_train, y_test = train_test_split(balanced_data[0], balanced_data[1], test_size=0.2, random_state=42)
            
            test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
            train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
            
            x_test = x_test[test_valid_rows]
            y_test = y_test[test_valid_rows]
            x_train = x_train[train_valid_rows]
            y_train = y_train[train_valid_rows]

            start_time = time.time()
            accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, classifiers[c], 0)["accuracy"]
            end_time = time.time()
            person_accuracies.append(accuracy)
            times_to_predict.append((end_time-start_time)/len(x_test))
            

    project.log("Classifier = {} | Accuracy = {} | Time: {}".format(type(classifiers[c]).__name__, st.mean(person_accuracies), st.mean(times_to_predict)), file="umafall_best_algorithm.log")
    
