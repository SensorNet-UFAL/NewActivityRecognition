# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from models.umafall_model import UMAFALL_Model
from models.arcma_model import ARCMA_Model
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
from tsfresh import extract_relevant_features
from scripts.save_workspace import save
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from pre_processing.balance_data import BalanceData
from pre_processing.get_accuracy import Get_Accuracy

#===INITIALIZATION===#
Debug.DEBUG = 0
processing = Processing_DB_Files()
project = Project()
s = save()

#===INIT BASES===#
hmp_person = "f1" 
umafall_person = 1
arcma_person = 15

models = []
models.append({"model_name":"hmp", "model":HMP_Model(), "person":hmp_person, "window":16, "where":""})
models.append({"model_name":"umafall", "model":UMAFALL_Model(), "person":umafall_person, "window":10, "where": "and body=2 and sensor=2"}) #MAGNETOMETER in WAIST
models.append({"model_name":"arcma", "model":ARCMA_Model(), "person":arcma_person, "window":26, "where":""})

####
classifier = ExtraTreesClassifier(n_estimators = 1000, random_state=1)

get_accuracy = Get_Accuracy()
balance_data = BalanceData()
threshold_balance_data = 40
for model in models:
    print(model['model_name'])
    data = model['model'].load_training_data_by_people(model['person'], additional_where = model['where'])
    accuracy_mean = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F-Score", "Time"])
    for w in range(10,110,10):
        try:
            print("Slicing Window....")
            data_tsfresh, y = model['model'].slice_by_window_tsfresh(data, w)
            y.index += 1
            del data_tsfresh["activity"]
            
            classes_counts = y.value_counts()
            if len(classes_counts) > 1:
                relevant_features = extract_relevant_features(data_tsfresh, y, column_id='id', column_sort='time')
                y = pd.DataFrame(y, columns=[model['model'].label_tag])
                
                balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
                if isinstance(balanced_data, tuple):
                    return_simple_accuracy = get_accuracy.get_accuracy_cross(classifier, balanced_data[0], balanced_data[1], n_to_split=5, threshold=0)
                    accuracy = return_simple_accuracy["accuracy"]
                    spent_time = return_simple_accuracy["spent_time"]
                    precision = return_simple_accuracy['precision']
                    recall = return_simple_accuracy['recall']
                    f_score = return_simple_accuracy['f-scores']
                    
                    out_aux = pd.DataFrame({"Window": w, "Classifier":[type(classifier).__name__], "Accuracy":[accuracy], "Precision":[precision], "Recall":[recall], "F-Score":[f_score], "Time":[spent_time]})
                    accuracy_mean = pd.concat([accuracy_mean, out_aux])
                
            del data_tsfresh
            del y
            del relevant_features
        except Exception as e:
            print(str(e))
    del data
    accuracy_mean.to_csv(s.path+"new_results{}{}_accuracy_by_window_{}.csv".format(slash, model['model_name'], type(classifier).__name__), sep='\t', encoding='utf-8')