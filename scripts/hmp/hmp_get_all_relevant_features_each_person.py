# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from tsfresh import extract_relevant_features
import pandas as pd
import numpy as np
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
s = save()
data_list_people = hmp.load_training_data_from_list_people(16, ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "m5", "m6", "m7", "f4", "m8", "m9", "f5", "m10", "m11"], remove_outliers=0.05) # Janela Fixa
#data_list_people = hmp.load_training_data_from_list_people(80, ["f1", "m1", "m2", "f2", "m3", "f3", "m4", "m5", "m6", "m7", "f4", "m8", "m9", "f5", "m10", "m11"], remove_outliers=0.05) # Janela encontrada


for p in data_list_people: 
    dataframe_1 = data_list_people[p]["training"]
    dataframe_2 = pd.DataFrame()
    labels = []
    id = 1
    for d in dataframe_1:
        if len(np.unique(d[hmp.label_tag])) < 2:
            d["id"] = pd.Series(np.full((1,d.shape[0]), id)[0], index=d.index)
            d["time"] = pd.Series(range((id-1)*d.shape[0], id*d.shape[0]), index=d.index)
            labels.append(d["activity"].iloc[0])
            dataframe_2 = dataframe_2.append(d, ignore_index=True)
            id = id + 1
    del dataframe_1  
    dataframe_3 = dataframe_2.drop(hmp.label_tag, 1)
    del dataframe_2
    y = np.array(labels)
    del labels
    y2 = pd.Series(y)
    del y

    y2.index+= 1
    classes_counts = y2["activity"].value_counts()
    if len(classes_counts) > 1:
        relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time')
        s.save_var(relevant_features, "hmp_relevant_features_fix_window{}relevant_features_{}.pkl".format(slash,p))
        s.save_var(y2, "hmp_relevant_features_fix_window{}y_{}.pkl".format(slash, p))
