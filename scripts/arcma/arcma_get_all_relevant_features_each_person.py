# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.arcma_model import ARCMA_Model
from tsfresh import extract_relevant_features
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
s = save()
window = 26 # Janela Fixa
persons = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#data_list_people = arcma.load_training_data_from_list_people(26, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], remove_outliers=0.05) # Janela Fixa
#data_list_people = arcma.load_training_data_from_list_people(40, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], remove_outliers=0.05) # Janela encontrada


for p in persons: 
    
    data = arcma.load_training_data_by_people(p)
    print("Slicing Window....")
    data_tsfresh, y = arcma.slice_by_window_tsfresh(data, window)
    y.index += 1
    del data_tsfresh["activity"]
    
    classes_counts = y.value_counts()
    if len(classes_counts) > 1:
        relevant_features = extract_relevant_features(data_tsfresh, y, column_id='id', column_sort='time')
        s.save_var(relevant_features, "arcma_relevant_features_fix_window{}relevant_features_{}.pkl".format(slash, p))
        s.save_var(y, "arcma_relevant_features_fix_window{}y_{}.pkl".format(slash, p))

