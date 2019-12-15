# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from tsfresh import extract_relevant_features
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
s = save()
#window = 16 # Janela Fixa
window = 50 # Melhor Janela
persons = ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "m5", "m6", "m7", "f4", "m8", "m9", "f5", "m10", "m11"]

for p in persons: 
    
    data = hmp.load_training_data_by_people(p)
    print("Slicing Window....")
    data_tsfresh, y = hmp.slice_by_window_tsfresh(data, window)
    y.index += 1
    del data_tsfresh["activity"]
    
    classes_counts = y.value_counts()
    if len(classes_counts) > 1:
        relevant_features = extract_relevant_features(data_tsfresh, y, column_id='id', column_sort='time')
        s.save_var(relevant_features, "hmp_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
        s.save_var(y, "hmp_relevant_features_best_window{}y_{}.pkl".format(slash, p))

