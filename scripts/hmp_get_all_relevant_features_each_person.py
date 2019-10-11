# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files
from sklearn.metrics import accuracy_score
from utils.project import Project
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, max_depth=1000, random_state=0) #Good performer
base_classification = Base_Classification(hmp, extra_trees)
s = save()
data_list_people = hmp.load_training_data_from_list_people(36, ["f1", "m1", "m2"], remove_outliers=0.05)


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

    relevant_features = extract_relevant_features(dataframe_3, y2, column_id='id', column_sort='time')
    s.save_var(relevant_features, "relevant_features_{}.pkl".format(p))
