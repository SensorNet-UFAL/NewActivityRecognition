# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
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
from pre_processing.balance_data import BalanceData
import statistics as st


#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
classifiers = {"Extratrees": ExtraTreesClassifier(n_estimators = 1000), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(n_estimators = 1000), "Decision Tree":tree.DecisionTreeClassifier(), "SVM":svm.SVC(probability=True)}
persons = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "m5", "m6", "m7", "f4", "m8", "m9", "f5", "m10", "m11"]
get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 40
#Select the best classifier
accuracy_mean = pd.DataFrame(columns=["Classifier", "Accuracy"])
project.log("=====================HMP_SELECT_BEST_ALGORITHM=====================", file="hmp_best_algorithm.log")
for c in classifiers:
    print(c)
    person_accuracies = []
    for p in persons:
        s = save()
        try:
            relevant_features = s.load_var("hmp_relevante_features_fix_window{}relevant_features_{}.pkl".format(slash, p))
            y = s.load_var("hmp_relevante_features_fix_window{}y_{}.pkl".format(slash, p))
            y = pd.DataFrame(y, columns=[hmp.label_tag])
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

            accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, classifiers[c], 0)["accuracy"]
            person_accuracies.append(accuracy)
            

    project.log("Classifier = {} | Accuracy = {}".format(type(classifiers[c]).__name__, st.mean(person_accuracies)), file="hmp_best_algorithm.log")
    
