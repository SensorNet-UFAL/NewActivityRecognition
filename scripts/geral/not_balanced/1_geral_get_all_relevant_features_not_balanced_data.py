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

#===INITIALIZATION===#
Debug.DEBUG = 0
processing = Processing_DB_Files()
project = Project()
s = save()

#===INIT BASES===#
hmp_persons = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "f4"] # at least 5 activities
#umafall_persons = [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17]
umafall_persons = [8,10,11,12,13,14,15,16,17]
arcma_persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

models = []
#Round 01
#models.append({"model_name":"hmp", "model":HMP_Model(), "persons":hmp_persons, "window":16, "where":""})
#models.append({"model_name":"umafall", "model":UMAFALL_Model(), "persons":umafall_persons, "window":10, "where": "and body=2 and sensor=2"}) #MAGNETOMETER in WAIST
#models.append({"model_name":"arcma", "model":ARCMA_Model(), "persons":arcma_persons, "window":26, "where":""})

#Round 02
models.append({"model_name":"hmp", "model":HMP_Model(), "persons":hmp_persons, "window":90, "where":""})
#models.append({"model_name":"umafall", "model":UMAFALL_Model(), "persons":umafall_persons, "window":10, "where": "and body=2 and sensor=2"}) #MAGNETOMETER in WAIST
models.append({"model_name":"arcma", "model":ARCMA_Model(), "persons":arcma_persons, "window":40, "where":""})

for model in models:
    for p in model["persons"]:
        try:
            data = model['model'].load_training_data_by_people(p, additional_where = model['where'])
            print("Slicing Window....")
            data_tsfresh, y = model['model'].slice_by_window_tsfresh(data, model["window"])
            y.index += 1
            del data_tsfresh["activity"]
            
            classes_counts = y.value_counts()
            if len(classes_counts) > 1:
                relevant_features = extract_relevant_features(data_tsfresh, y, column_id='id', column_sort='time')
                s.save_var(relevant_features, "new_features{}{}_relevant_features_window_{}{}relevant_features_{}.pkl".format(slash, model['model_name'], model['window'], slash, p))
                s.save_var(y, "new_features{}{}_relevant_features_window_{}{}y_{}.pkl".format(slash, model['model_name'], model['window'], slash, p))
            del data
            del data_tsfresh
            del y
            del relevant_features
        except Exception as e:
            print(str(e))
            