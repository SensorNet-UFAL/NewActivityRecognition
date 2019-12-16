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
from pre_processing.balance_data import BalanceData


#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 10000, random_state=0)
base_classification = Base_Classification(arcma, extra_trees)
balance_data = BalanceData()
threshold_balance_data = 40

#===LOAD FEATURES===#

#Interate threshold to find de best value#
s = save()
person_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
accuracy_threshould_list = []
data = {}
threshold = 0.60
project.log("=========== ARCMA Outlier Accuracy, Thresold = {}===========".format(threshold),file="arcma_log_outlier_accuracy.log")
for p in person_list:
    project.log("===========Person {}===========".format(p), file="arcma_log_outlier_accuracy.log")
    data = s.load_var("arcma_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("arcma_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[arcma.label_tag])
     
    print("------------------------------------")
    print("Person: {}".format(p))
    print("------------------------------------")
    
    balanced_data = balance_data.balance_data(data, y, threshold_balance_data)
    if isinstance(balanced_data, tuple):
        return_accuracy = base_classification.get_accuracy.stratified_kfold_accuracy_outlier(balanced_data[0], balanced_data[1], extra_trees, threshold, p)
        project.log(str(return_accuracy), file="arcma_log_outlier_accuracy.log")

