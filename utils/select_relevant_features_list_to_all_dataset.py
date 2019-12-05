# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:57:55 2019

@author: wylken.machado
"""
from scripts.save_workspace import save
from utils.project import slash

s = save()


relevant_features = s.load_var("arcma_relevante_features{}relevant_features_{}.pkl".format(slash, "1"))
arcma_features = list(relevant_features.columns.values)
relevant_features = s.load_var("hmp_relevante_features{}relevant_features_{}.pkl".format(slash, "f1"))
hmp_features = list(relevant_features.columns.values)
relevant_features = s.load_var("umafall_relevante_features{}relevant_features_{}.pkl".format(slash, "1"))
umafall_features = list(relevant_features.columns.values)
