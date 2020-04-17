# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
#===== Machine Learn =====#
from sklearn.neural_network import MLPClassifier #multi-layer percept
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
t_aux = []
for i in range(0,2):
    t_aux.append(500)
t = tuple(t_aux)
classifiers = {"MLP": MLPClassifier(random_state=1, solver="adam", activation="relu", max_iter=100000, alpha=1e-5, hidden_layer_sizes=t)}
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
    person_f_score = []
    person_precision = []
    person_recall = []
    times_to_predict = []
    for p in persons:
        s = save()
        try:
            relevant_features = s.load_var("hmp_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
            y = s.load_var("hmp_relevant_features_best_window{}y_{}.pkl".format(slash, p))
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

            return_simple_accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, classifiers[c], 0)
            accuracy = return_simple_accuracy["accuracy"]
            spent_time = return_simple_accuracy["spent_time"]
            metrics = return_simple_accuracy["metrics"]
            person_accuracies.append(accuracy)
            times_to_predict.append(spent_time)
            person_precision.append(metrics[0])
            person_recall.append(metrics[1])
            person_f_score.append(metrics[2])
            
    project.log("Classifier = {} | Accuracy = {} | Precision: {} | Recall: {} | F-Score: {} | Time: {}".format(type(classifiers[c]).__name__, st.mean(person_accuracies), st.mean(person_precision), st.mean(person_recall), st.mean(person_f_score),st.mean(times_to_predict)), file="new_results{}hmp_best_algorithm.log".format(slash))
    
