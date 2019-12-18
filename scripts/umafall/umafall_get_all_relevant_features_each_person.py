# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.umafall_model import UMAFALL_Model
from tsfresh import extract_relevant_features
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
umafall = UMAFALL_Model()
processing = Processing_DB_Files()
project = Project()
s = save()
#window = 10 # Janela Fixa
window = 30 # Melhor Janela
persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#persons = [7,8,9,10,11,12,13,14,15]

for p in persons: 
    
    data = umafall.load_training_data_by_people(p, additional_where = "and body=2 and sensor=2") #MAGNETOMETER in WAIST
    print("Slicing Window....")
    data_tsfresh, y = umafall.slice_by_window_tsfresh(data, window)
    if data_tsfresh is None:
        continue
    
    y.index += 1
    del data_tsfresh["activity"]
    
    classes_counts = y.value_counts()
    if len(classes_counts) > 1:
        relevant_features = extract_relevant_features(data_tsfresh, y, column_id='id', column_sort='time')
        s.save_var(relevant_features, "umafall_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
        s.save_var(y, "umafall_relevant_features_best_window{}y_{}.pkl".format(slash, p))

