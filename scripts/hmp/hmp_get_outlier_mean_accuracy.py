# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project
from scripts.save_workspace import save

#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 10000, random_state=0)
base_classification = Base_Classification(hmp, extra_trees)

#===LOAD FEATURES===#

#Interate threshold to find de best value#

s = save()
relevant_features = s.load_var("hmp_relevant_features\\relevant_features_f1.pkl")
y = s.load_var("hmp_relevant_features\\y_f1.pkl")
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

#return_dataframe, return_accuracy, data_from_each_person = base_classification.get_accuracy.get_outliers_confused_with_activities(data, hmp, extra_trees,0.55)
accuracy_threshould = pd.DataFrame(columns=["accuracy","discarted", "len_activity"])    

threshold = 0.55

return_proba = base_classification.get_accuracy.simple_accuracy_mean_to_each_person_with_proba(data, hmp, extra_trees, threshold)
return_proba = return_proba[list(return_proba.keys())[0]]
accuracy_threshould = accuracy_threshould.append(return_proba, ignore_index=True)
print(accuracy_threshould)

