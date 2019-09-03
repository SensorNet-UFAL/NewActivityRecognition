# -*- coding: utf-8 -*-
from models.model import Model
import pandas as pd
from outlier.outlier_commons import Outlier_Commons


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
        valid_data_from_each_person = {}
        for p in data_from_each_person:
            #Get data of person
            person_data = data_from_each_person[p]
            #Getting unique labels to the data
            clf.classes_ = person_data["test"]["test_labels"].activity.unique()
            #Fit model
            clf.fit(person_data['training']['training_features'], person_data['training']['training_labels'])
            #Getting the probability table
            pred = clf.predict_proba(person_data['test']['test_features'])
            pred = pd.DataFrame(pred, columns = clf.classes_)
            #Filtering data with probability greater than the threshold
            valid_indexes = self.get_indexes_with_valid_predictions(pred, threshold)
            valid_data_from_each_person[p] = {"training":{}, "test":{}}
            valid_data_from_each_person[p]["training"]["training_features"] = person_data['training']['training_features'].iloc[valid_indexes,:]
            valid_data_from_each_person[p]["training"]["training_labels"] = person_data['training']['training_labels'].iloc[valid_indexes,:]
            valid_data_from_each_person[p]["test"]["test_features"] = person_data['test']['test_features'].iloc[valid_indexes,:]
            valid_data_from_each_person[p]["test"]["test_labels"] = person_data['test']['test_labels'].iloc[valid_indexes,:]
            
            #clf.fit(valid_data_from_each_person[p]['training']['training_features'], valid_data_from_each_person[p]['training']['training_labels'])
            accuracies[p] = clf.score(valid_data_from_each_person[p]['test']['test_features'], valid_data_from_each_person[p]['test']['test_labels'])
            
        return accuracies
    
    def get_indexes_with_valid_predictions(self, dataframe_predicions:pd.DataFrame, threshold):
        return_indexes = []
        for index, row in dataframe_predicions.iterrows():
            for i, value in enumerate(row):
                if value > threshold:
                    return_indexes.append(index)
                    break
        return return_indexes
            
    def simple_accuracy_outlier_activity(self, data_from_each_person, model:Model, clf, activity, threshold):
        outliers_commons = Outlier_Commons()    
        for p in data_from_each_person:
                person_data = data_from_each_person[p]
                training = person_data['training']['training_features']
                training_labels = person_data['training']['training_labels']
                test = person_data['test']['test_features']
                test_labels = person_data['test']['test_labels']
                training, training_labels, test, test_labels, outlier, outlier_labels = outliers_commons.generate_outliers(training, training_labels, test, test_labels, activity)
                return training_labels, test_labels, outlier_labels