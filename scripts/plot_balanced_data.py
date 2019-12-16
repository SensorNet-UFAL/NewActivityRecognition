# -*- coding: utf-8 -*-
# IMPORTS #
from models.arcma_model import ARCMA_Model
from models.hmp_model import HMP_Model
from models.umafall_model import UMAFALL_Model
import pandas as pd
from utils.project import slash
from scripts.save_workspace import save
from pre_processing.balance_data import BalanceData
import time

s = save()
balance_data = BalanceData()
threshold_balance_data = 40

def plot_hist(features, y, text):
    classes_counts = y["activity"].value_counts()
    classes_counts.plot(kind='bar', title=text, figsize=(18.5, 10.5))


#====ARCMA====#
def plot_arcma():
    arcma = ARCMA_Model()
    p=15
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
    #plot_hist(relevant_features, y, 'Base ARCMA desbalanceada.')
    plot_hist(balanced_data[0], balanced_data[1], 'Base ARCMA Balanceada.')

#====HMP====#
def plot_hmp():
    hmp = HMP_Model()
    p="f1"
    relevant_features = s.load_var("hmp_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("hmp_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[hmp.label_tag])
    balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
    #plot_hist(relevant_features, y, 'Base HMP desbalanceada.')
    plot_hist(balanced_data[0], balanced_data[1], 'Base HMP Balanceada.')
    
#====UMAFALL====#
def plot_umafall():
    umafall = UMAFALL_Model()
    p=1
    relevant_features = s.load_var("umafall_relevant_features_best_window{}relevant_features_{}.pkl".format(slash, p))
    y = s.load_var("umafall_relevant_features_best_window{}y_{}.pkl".format(slash, p))
    y = pd.DataFrame(y, columns=[umafall.label_tag])
    balanced_data = balance_data.balance_data(relevant_features, y, threshold_balance_data)
    #plot_hist(relevant_features, y, 'Base HMP desbalanceada.')
    plot_hist(balanced_data[0], balanced_data[1], 'Base HMP Balanceada.')


plot_umafall()