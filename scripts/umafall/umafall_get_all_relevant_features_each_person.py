# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.umafall_model import UMAFALL_Model
from tsfresh import extract_relevant_features
import pandas as pd
import numpy as np
from pre_processing.processing_db_files import Processing_DB_Files
from utils.project import Project, slash
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
umafall = UMAFALL_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, max_depth=1000, random_state=0) #Good performer
s = save()
p_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
p_list=[1]
data_list_people = umafall.load_training_data_from_list_people(10, p_list, remove_outliers=0.05, additional_where="and sensor=2 and body=3") # select magnetometer in waist - best accuracy


for p in data_list_people: 
    dataframe_1 = data_list_people[p]["training"]
    dataframe_2 = pd.DataFrame()
    labels = []
    id = 1
    for d in dataframe_1:
        if len(np.unique(d[umafall.label_tag])) < 2:
            d["id"] = pd.Series(np.full((1,d.shape[0]), id)[0], index=d.index)
            d["time"] = pd.Series(range((id-1)*d.shape[0], id*d.shape[0]), index=d.index)
            labels.append(d["activity"].iloc[0])
            dataframe_2 = dataframe_2.append(d, ignore_index=True)
            id = id + 1
    del dataframe_1  
    dataframe_3 = dataframe_2.drop(umafall.label_tag, 1)
    del dataframe_2
    y = np.array(labels)
    del labels
    y2 = pd.Series(y)
    del y

    y2.index+= 1

    relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time')
    s.save_var(relevant_features, "umafall{}relevant_features_{}.pkl".format(slash, p))
    s.save_var(y2, "umafall{}y_{}.pkl".format(slash, p))

