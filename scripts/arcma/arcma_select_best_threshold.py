# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.arcma_model import ARCMA_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project
from scripts.save_workspace import save

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
relevant_features = s.load_var("arcma_relevante_features\\relevant_features_1.pkl")
y = s.load_var("arcma_relevante_features\\y_1.pkl")
y = pd.DataFrame(y, columns=[arcma.label_tag])
X_train, X_test, y_train, y_test = train_test_split(relevant_features, y, test_size=0.2, random_state=42)
data = {}
data["1"] = {}
data["1"]["training"] = {}
data["1"]["training"]["training_features"] = X_train
data["1"]["training"]["training_labels"] = y_train

data["1"]["test"] = {}
data["1"]["test"]["test_features"] = X_test
data["1"]["test"]["test_labels"] = y_test

#return_dataframe, return_accuracy, data_from_each_person = base_classification.get_accuracy.get_outliers_confused_with_activities(data, hmp, extra_trees,0.55)
accuracy_threshould = pd.DataFrame(columns=["accuracy","discarted", "len_activity", "threshold"])
for t in np.arange(0.05, 1, 0.05):    
    return_proba = base_classification.get_accuracy.simple_accuracy_mean_to_each_person_with_proba(data, arcma, extra_trees, t)
    return_proba = return_proba[list(return_proba.keys())[0]]
    return_proba["threshold"] = t
    accuracy_threshould = accuracy_threshould.append(return_proba, ignore_index=True)

