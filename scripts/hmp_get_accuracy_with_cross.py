# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
import pandas as pd
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project
from scripts.save_workspace import save
from pre_processing.get_accuracy import Get_Accuracy
from sklearn.model_selection import KFold
import numpy as np

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 10000, random_state=0)
get_accuracy = Get_Accuracy()

#===LOAD FEATURES===#

#Interate threshold to find de best value#

s = save()
relevant_features = s.load_var("hmp_relevant_features\\relevant_features_f1.pkl")
y = s.load_var("hmp_relevant_features\\y_f1.pkl")
y = pd.DataFrame(y, columns=[hmp.label_tag])
kf = KFold(n_splits=3)


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]