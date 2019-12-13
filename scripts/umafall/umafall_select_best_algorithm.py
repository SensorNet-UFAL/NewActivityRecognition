# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.umafall_model import UMAFALL_Model
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
umafall = UMAFALL_Model()
processing = Processing_DB_Files()
project = Project()
classifiers = {"Extratrees": ExtraTreesClassifier(n_estimators = 1000), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(n_estimators = 1000), "Decision Tree":tree.DecisionTreeClassifier(), "SVM":svm.SVC(probability=True), "MPL":MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)}
persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 100
#Select the best classifier
accuracy_mean = pd.DataFrame(columns=["Classifier", "Accuracy"])
project.log("=====================UMAFALL_SELECT_BEST_ALGORITHM=====================", file="umafall_best_algorithm.log")
for c in classifiers:
    print(c)
    person_accuracies = []
    for p in persons:
        s = save()
        try:
            relevant_features = s.load_var("umafall_relevante_features_fix_window{}relevant_features_{}.pkl".format(slash, p))
            y = s.load_var("umafall_relevante_features_fix_window{}y_{}.pkl".format(slash, p))
            y = pd.DataFrame(y, columns=[umafall.label_tag])
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
            break

    project.log("Classifier = {} | Accuracy = {}".format(type(classifiers[c]).__name__, st.mean(person_accuracies)), file="umafall_best_algorithm.log")
    break
