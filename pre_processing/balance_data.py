# -*- coding: utf-8 -*-

import pandas as pd

class BalanceData(object):
    def __init__(self):
        super().__init__()
        
    def balance_data(self, features, y, threshould = 0, y_tag = "activity"):
        classes_counts = y[y_tag].value_counts()
        invalid_classes = list()
        #procurando valores abaixo do threshold
        for index, value in classes_counts.iteritems():
            if value < threshould:
                invalid_classes.append(index)
        #Retirando os valores abaixo do threshoud
        invalid_indexes = []
        for i in invalid_classes:
            invalid_indexes = invalid_indexes + list(y[y[y_tag]==i].index)
        y_new = y.copy()
        y_new = y_new.drop(labels=invalid_indexes)
        classes_counts = y_new[y_tag].value_counts()
        if len(classes_counts) > 1:
            samples = list()
            for classe, value in classes_counts.iteritems():
                samples.append(y_new[y_new[y_tag]==classe].sample(classes_counts.min()))
            y_new = pd.concat(samples)
            features_new = features.iloc[y_new.index-1,:]
            return features_new, y_new
        else:
            return None