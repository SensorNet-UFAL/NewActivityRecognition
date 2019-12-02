# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.arcma_model import ARCMA_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project, slash
from scripts.save_workspace import save
from sklearn.model_selection import StratifiedKFold
import numpy as np


#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 10000, random_state=0)
base_classification = Base_Classification(arcma, extra_trees)

#===LOAD FEATURES===#

#Interate threshold to find de best value#
s = save()
person_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
accuracy_threshould_list = []
data = {}
threshold = 0.80
for p in person_list:
    data = s.load_var("arcma_relevante_features{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("arcma_relevante_features{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[arcma.label_tag])
     
    print("------------------------------------")
    print("Person: {}".format(p))
    print("------------------------------------")
    
    return_accuracy = base_classification.get_accuracy.stratified_kfold_accuracy_outlier(data, y, extra_trees, 0.75, p)
    accuracy_threshould_list.append(return_accuracy)

