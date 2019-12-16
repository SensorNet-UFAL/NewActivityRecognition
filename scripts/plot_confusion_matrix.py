# -*- coding: utf-8 -*-
# IMPORTS #
from models.arcma_model import ARCMA_Model
from models.hmp_model import HMP_Model
from models.umafall_model import UMAFALL_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
from utils.project import slash
from scripts.save_workspace import save
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from pre_processing.balance_data import BalanceData
s = save()
balance_data = BalanceData()
threshold_balance_data = 40


def plot_confusion_matrix(model, relevant_features_new, y_new, threshold_classification):
    
    extra_trees = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
    base_classification = Base_Classification(model, extra_trees)
    
    #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    sss = StratifiedKFold(n_splits=3, shuffle=False, random_state=10)
    for train_index, test_index in sss.split(relevant_features_new, y_new):
        x_train, x_test = relevant_features_new.iloc[train_index,:], relevant_features_new.iloc[test_index,:]
        y_train, y_test = y_new.iloc[train_index,:], y_new.iloc[test_index,:]
        break
    
    #x_train, x_test, y_train, y_test = train_test_split(relevant_features_new, y_new, test_size=0.3, random_state=42)
    extra_trees.fit(x_train, y_train)
    pred = extra_trees.predict_proba(x_test)
    pred = pd.DataFrame(pred, columns = extra_trees.classes_)
    valid_indexes = base_classification.get_accuracy.get_indexes_with_valid_predictions(pred, threshold_classification)
    
    x_test_valid = x_test.iloc[valid_indexes,:]
    y_test_valid = y_test.iloc[valid_indexes,:]
    
    base_classification.get_accuracy.plot_confusion_matrix(x_test_valid, y_test_valid, extra_trees)
    print("Accuracy => {}".format(extra_trees.score(x_test_valid, y_test_valid)))
    base_classification.get_accuracy.plot_confusion_matrix(x_test, y_test, extra_trees)
    print("Accuracy => {}".format(extra_trees.score(x_test, y_test)))

#====ARCMA====#
def plot_arcma():
    arcma = ARCMA_Model()
    p=15
    arcma_threshold_classification = 0.65
    relevant_features = s.load_var("arcma_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("arcma_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[arcma.label_tag])
    #Remove Activit with label 0, not discribed in arcma doc 
    #invalid_indexes = list(y[y["activity"]==0].index)
    #y_new = y.drop(labels=invalid_indexes)
    #relevant_features_new = relevant_features.drop(labels=invalid_indexes)
    #Translate Arcma Labels
    y[y["activity"] == 0] = "Indefinida"
    y[y["activity"] == 1] = "working_at_computer"
    y[y["activity"] == 2] = "standing_up_walking_going_updown_stairs"
    y[y["activity"] == 3] = "standing"
    y[y["activity"] == 4] = "walking"
    y[y["activity"] == 5] = "going_updown_stairs"
    y[y["activity"] == 6] = "walking_and_talking_with_someone."
    y[y["activity"] == 7] = "talking_while_standing"
    
    balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
    plot_confusion_matrix(arcma, balanced_data[0], balanced_data[1], arcma_threshold_classification)


#====HMP====#
def plot_hmp():
    hmp = HMP_Model()
    p="f1"
    hmp_threshold_classification = 0.65
    relevant_features = s.load_var("hmp_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("hmp_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[hmp.label_tag])
    balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
    plot_confusion_matrix(hmp, balanced_data[0], balanced_data[1], hmp_threshold_classification)

def umafall():
    umafall = UMAFALL_Model()
    p=1
    umafall_threshold_classification = 0.65
    relevant_features = s.load_var("umafall_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("umafall_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[umafall.label_tag])
    balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
    plot_confusion_matrix(umafall, balanced_data[0], balanced_data[1], umafall_threshold_classification)
    
#plot_hmp()
plot_arcma()





