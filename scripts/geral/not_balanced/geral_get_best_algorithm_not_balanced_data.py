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
import statistics as st

#===INITIALIZATION===#
Debug.DEBUG = 0
processing = Processing_DB_Files()
project = Project()

#===INIT BASES===#
hmp_persons = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "f4"] # at least 5 activities
umafall_persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
arcma_persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
models = [{"model_name":"hmp", "model":HMP_Model(), "persons":hmp_persons}]
models.append({"model_name":"umafall", "model":UMAFALL_Model(), "persons":umafall_persons})
models.append({"model_name":"arcma", "model":ARCMA_Model(), "persons":arcma_persons})


#tuple from MPL
t_aux = []
for i in range(0,500):
    t_aux.append(500)
t = tuple(t_aux)
####
classifiers = {"MPL": MLPClassifier(random_state=1, solver="adam", activation="relu", max_iter=100000, alpha=1e-5, hidden_layer_sizes=t), "Extratrees": ExtraTreesClassifier(n_estimators = 1000, random_state=1), "Knn":KNeighborsClassifier(n_neighbors=5), "Naive Bayes":GaussianNB(), "RandomForest":RandomForestClassifier(n_estimators = 1000, random_state=1), "Decision Tree":tree.DecisionTreeClassifier(random_state=1), "SVM":svm.SVC(probability=True, random_state=1)}

get_accuracy = Get_Accuracy()
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
            try:
                relevant_features = s.load_var("{}_relevant_features_best_window{}relevant_features_{}.pkl".format(model['model_name'], slash, p))
                y = s.load_var("{}_relevant_features_best_window{}y_{}.pkl".format(model['model_name'], slash, p))
                y = pd.DataFrame(y, columns=[model['model'].label_tag])
            except:
                print("file from person {} not found!".format(p))
                continue
            
            
            #x_train, x_test, y_train, y_test = train_test_split(relevant_features, y, test_size=0.2, random_state=42)
            
            #test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
            #train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
            
            #x_test = x_test[test_valid_rows]
            #y_test = y_test[test_valid_rows]
            #x_train = x_train[train_valid_rows]
            #y_train = y_train[train_valid_rows]

            return_simple_accuracy = get_accuracy.get_accuracy_cross(classifiers[c], relevant_features, y, n_to_split=5, threshold=0)
            accuracy = return_simple_accuracy["accuracy"]
            spent_time = return_simple_accuracy["spent_time"]
            person_accuracies.append(accuracy)
            times_to_predict.append(spent_time)
            person_precision.append(return_simple_accuracy['precision'])
            person_recall.append(return_simple_accuracy['recall'])
            person_f_score.append(return_simple_accuracy['f-scores'])
                
        out_aux = pd.DataFrame({"Classifier":[type(classifiers[c]).__name__], "Accuracy":[st.mean(person_accuracies)], "Precision":[st.mean(person_precision)], "Recall":[st.mean(person_recall)], "F-Score":[st.mean(person_f_score)], "Time":[st.mean(times_to_predict)]})
        accuracy_mean = pd.concat([accuracy_mean, out_aux])
        #project.log("Classifier = {} | Accuracy = {} | Precision: {} | Recall: {} | F-Score: {} | Time: {}".format(, , , , ,), file="new_results{}hmp_best_algorithm.log".format(slash))

    accuracy_mean.to_csv(s.path+"new_results{}{}_best_algorithm_not_balanced.csv".format(slash, model['model_name']), sep='\t', encoding='utf-8')
        
    
