# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.arcma_model import ARCMA_Model
from classifiers.base_classification import Base_Classification
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
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


#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
classifiers = {"Extratrees": ExtraTreesClassifier(n_estimators = 1000, random_state=0), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(random_state=1), "Decision Tree":tree.DecisionTreeClassifier(), "SVM":svm.SVC(probability=True), "MPL":MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)}
persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
get_accuracy = Get_Accuracy()
#Select the best classifier
accuracy_mean = pd.DataFrame(columns=["Classifier", "Accuracy"])
project.log("=====================ARCMA_SELECT_BEST_ALGORITHM=====================")
for c in classifiers:
    print(c)
    #base_classification = Base_Classification(arcma, classifiers[key])
    #_, accuracies_proba = base_classification.predict_for_list_people_with_proba(36, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],0.1)
    #mean_aux = []
    person_accuracies = []
    for p in persons:
        s = save()
        relevant_features = s.load_var("arcma_relevant_features{}relevant_features_{}.pkl".format(slash, p))
        y = s.load_var("arcma_relevant_features{}y_{}.pkl".format(slash, p))
        y = pd.DataFrame(y, columns=[arcma.label_tag])
        
        x_train, x_test, y_train, y_test = train_test_split(relevant_features, y, test_size=0.2, random_state=42)
        
        test_valid_rows = np.isfinite(x_test["y__quantile__q_0.8"])
        train_valid_rows = np.isfinite(x_train["y__quantile__q_0.8"])
        
        x_test = x_test[test_valid_rows]
        y_test = y_test[test_valid_rows]
        x_train = x_train[train_valid_rows]
        y_train = y_train[train_valid_rows]
        
        accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, classifiers[c], 0)["accuracy"]
        person_accuracies.append(accuracy)
    
    #accuracy_mean = accuracy_mean.append(pd.DataFrame([[type(classifiers[c]).__name__, st.mean(person_accuracies)]], columns=["Classifier", "Accuracy"]))
    
    project.log("Classifier = {} | Accuracy = {}".format(type(classifiers[c]).__name__, st.mean(person_accuracies)))

#base_classification = Base_Classification(arcma, ExtraTreesClassifier(n_estimators = 1000, random_state=0)) # Best algoritm
#best_window = base_classification.find_best_window(range(2, 100, 2), [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#dataframe = pd.DataFrame(best_window)

