# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
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

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
classifiers = {"Extratrees": ExtraTreesClassifier(n_estimators = 1000, random_state=0), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(random_state=1), "Decision Tree":tree.DecisionTreeClassifier(), "SVM":svm.SVC(probability=True), "MPL":MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)}

#Select the best classifier
accuracy_mean = {}
for key in classifiers:
    print(key)
    base_classification = Base_Classification(hmp, classifiers[key])
    _, accuracies_proba = base_classification.predict_for_list_people_with_proba(50, ["f1","m1","m2"],0.1)
    mean_aux = []
    for a in accuracies_proba:
        mean_aux.append(accuracies_proba[a]["accuracy"])
    accuracy_mean[key] = mean(mean_aux)

#Select de best windows
base_classification = Base_Classification(hmp, ExtraTreesClassifier(n_estimators = 1000, random_state=0)) # Best algoritm
best_window = base_classification.find_best_window(range(2, 100, 2), ["f1","m1","m2"])

