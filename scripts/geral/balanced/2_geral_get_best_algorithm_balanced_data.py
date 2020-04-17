# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from models.umafall_model import UMAFALL_Model
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
from sklearn.neural_network import MLPClassifier #multi-layer percept
import pandas as pd
from pre_processing.get_accuracy import Get_Accuracy
from scripts.save_workspace import save
from pre_processing.balance_data import BalanceData
import statistics as st

#===INITIALIZATION===#
Debug.DEBUG = 0
processing = Processing_DB_Files()
project = Project()

#===INIT BASES===#
hmp_persons = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "f4"] # at least 5 activities
umafall_persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
arcma_persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
models = []
models.append({"model_name":"hmp", "model":HMP_Model(), "persons":hmp_persons, "window":16})
models.append({"model_name":"umafall", "model":UMAFALL_Model(), "persons":umafall_persons, "window":10})
models.append({"model_name":"arcma", "model":ARCMA_Model(), "persons":arcma_persons, "window":26})


#tuple from MPL
t_aux = []
for i in range(0,500):
    t_aux.append(500)
t = tuple(t_aux)
####
classifiers = {"MPL": MLPClassifier(random_state=1, solver="adam", activation="relu", max_iter=100000, alpha=1e-5, hidden_layer_sizes=t), "Extratrees": ExtraTreesClassifier(n_estimators = 1000, random_state=1), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(n_estimators = 1000, random_state=1), "Decision Tree":tree.DecisionTreeClassifier(random_state=1), "SVM":svm.SVC(probability=True, random_state=1)}


get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 40
#Select the best classifier
for model in models:
    print(model['model_name'])
    persons = model['persons']
    accuracy_mean = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F-Score", "Time"])
    print("====================={}_SELECT_BEST_ALGORITHM=====================".format(model['model_name']))
    for c in classifiers:
        print(c)
        person_accuracies = []
        person_f_score = []
        person_precision = []
        person_recall = []
        times_to_predict = []
        for p in persons:
            s = save()
            print("####### PERSON ##### ===> {}".format(p))
            try:
                relevant_features_path =  "new_features{}{}_relevant_features_window_{}{}relevant_features_{}.pkl".format(slash, model['model_name'], model['window'], slash, p)
                y_path = "new_features{}{}_relevant_features_window_{}{}y_{}.pkl".format(slash, model['model_name'], model['window'], slash, p)
                print(relevant_features_path)
                print(y_path)
                relevant_features = s.load_var(relevant_features_path)
                y = s.load_var(y_path)
                y = pd.DataFrame(y, columns=[model['model'].label_tag])
            except:
                print("file from person {} not found!".format(p))
                continue
            
            
            balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
            
            if isinstance(balanced_data, tuple):
                
                relevant_features = balanced_data[0]
                y = balanced_data[1]
            
                relevant_features.index = pd.RangeIndex(len(relevant_features.index))
                y.index = pd.RangeIndex(len(y.index))
            
                #x_train, x_test, y_train, y_test = train_test_split(balanced_data[0], balanced_data[1], test_size=0.2, random_state=42)
                
                #test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
                #train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
                
                #x_test = x_test[test_valid_rows]
                #y_test = y_test[test_valid_rows]
                #x_train = x_train[train_valid_rows]
                #y_train = y_train[train_valid_rows]
    
                #return_simple_accuracy = get_accuracy.simple_accuracy_with_valid_predictions(x_train, x_test, y_train, y_test, classifiers[c], 0)
                #try:
                print("############## PERSON =>>>>>> {}".format(p))
                return_simple_accuracy = get_accuracy.get_accuracy_cross(classifiers[c], relevant_features, y, n_to_split=5, threshold=0)
                accuracy = return_simple_accuracy["accuracy"]
                spent_time = return_simple_accuracy["spent_time"]
                person_accuracies.append(accuracy)
                times_to_predict.append(spent_time)
                person_precision.append(return_simple_accuracy['precision'])
                person_recall.append(return_simple_accuracy['recall'])
                person_f_score.append(return_simple_accuracy['f-scores'])
                #except Exception as e:
                #    print("Error to get accuracy person {}! Error: {}".format(p, str(e)))
                #    continue
        out_aux = pd.DataFrame({"Classifier":[type(classifiers[c]).__name__], "Accuracy":[st.mean(person_accuracies)], "Precision":[st.mean(person_precision)], "Recall":[st.mean(person_recall)], "F-Score":[st.mean(person_f_score)], "Time":[st.mean(times_to_predict)]})
        accuracy_mean = pd.concat([accuracy_mean, out_aux])
        #project.log("Classifier = {} | Accuracy = {} | Precision: {} | Recall: {} | F-Score: {} | Time: {}".format(, , , , ,), file="new_results{}hmp_best_algorithm.log".format(slash))

    accuracy_mean.to_csv(s.path+"new_results{}{}_best_algorithm_balanced_data_window{}.csv".format(slash, model['model_name'], model['window']), sep='\t', encoding='utf-8')
        
    
