# -*- coding: utf-8 -*-
from models.hmp_model import HMP_Model
#===== Machine Learn =====#
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn import svm #SVM
from sklearn.neural_network import MLPClassifier #multi-layer perceptro
#==== UTILS ====#
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing.get_accuracy import Get_Accuracy
import numpy as np
import statistics as st
from utils.project import Project, slash
from scripts.save_workspace import save

#===INITIALIZATION===#
hmp = HMP_Model()
classifiers = {"Extratrees": ExtraTreesClassifier(n_estimators = 1000, random_state=0), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(random_state=1), "Decision Tree":tree.DecisionTreeClassifier(), "SVM":svm.SVC(probability=True), "MPL":MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)}
persons = ["f1","m1","m2"]
get_accuracy = Get_Accuracy()
project = Project()

project.log("=====================HMP_SELECT_BEST_ALGORITHM=====================")
for c in classifiers:
    print(c)

    person_accuracies = []
    for p in persons:
        s = save()
        relevant_features = s.load_var("hmp_relevant_features{}relevant_features_{}.pkl".format(slash, p))
        y = s.load_var("hmp_relevant_features{}y_{}.pkl".format(slash, p))
        y = pd.DataFrame(y, columns=[hmp.label_tag])
        
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
