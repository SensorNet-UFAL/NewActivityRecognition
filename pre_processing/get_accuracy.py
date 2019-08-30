# -*- coding: utf-8 -*-
from models.model import Model


class Get_Accuracy(object):
    
    def __init__(self):
        super().__init__()
        
    def simple_accuracy_mean_to_each_person(self, data_from_each_person, model:Model, clf):
        accuracies = {}
        for p in data_from_each_person:
            clf.fit(data_from_each_person[p]['training']['training_features'], data_from_each_person[p]['training']['training_labels'])
            accuracies[p] = clf.score(data_from_each_person[p]['test']['test_features'], data_from_each_person[p]['test']['test_labels'])
        return accuracies
    def simple_accuracy_mean_to_each_person_with_proba(self, data_from_each_person, model:Model, clf, threshold):
        accuracies = {}
        for p in data_from_each_person:
            clf.classes_ = data_from_each_person[p]["test"]["test_labels"].activity.unique()
            clf.fit(data_from_each_person[p]['training']['training_features'], data_from_each_person[p]['training']['training_labels'])
            pred = clf.predict_proba(data_from_each_person[p]['test']['test_features'])
            pred2 = clf.predict(data_from_each_person[p]['test']['test_features'])
            print("PERSON => {}".format(p))
            return pred, pred2, data_from_each_person[p]
        return accuracies
        
        

