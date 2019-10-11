# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project
from scripts.save_workspace import save
import pandas as pd
from pre_processing.get_accuracy import Get_Accuracy

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
relevant_features = s.load_var("Relevant_features.pkl")
y = s.load_var("y.pkl")
y = pd.DataFrame(y, columns=[hmp.label_tag])
X_train, X_test, y_train, y_test = train_test_split(relevant_features, y, test_size=0.2, random_state=42)
data = {}
data["f1"] = {}
data["f1"]["training"] = {}
data["f1"]["training"]["training_features"] = X_train
data["f1"]["training"]["training_labels"] = y_train

data["f1"]["test"] = {}
data["f1"]["test"]["test_features"] = X_test
data["f1"]["test"]["test_labels"] = y_test

return_dataframe, return_accuracy, data_from_each_person = get_accuracy.get_outliers_confused_with_activities(data, extra_trees, 0.6)
