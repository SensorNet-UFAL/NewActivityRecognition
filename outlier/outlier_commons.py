# -*- coding: utf-8 -*-

import numpy as np
from pre_processing.processing_db_files import Processing_DB_Files
from models.model import Model
from utils.debug import Debug

class Outlier_Commons(object):
    
    def __init__(self):
        super().__init__()
    
    def outlier_prepare(self, model:Model, people:str, windows_size:int, activity_outlier:str):
        processing = Processing_DB_Files()
        model.load_training_data_by_window_by_people(people, windows_size)
        training, training_labels, test, test_labels = processing.calculating_features(model)
        return self.generate_outliers(training, training_labels, test, test_labels, activity_outlier)

    def get_indexes(self, training_labels, test_labels, activity):
        train_indexes = np.where(training_labels["activity"] != activity)[0]
        test_indexes = np.where(test_labels["activity"] != activity)[0]
        outliers_indexes = np.where(training_labels["activity"] == activity)[0]
        return train_indexes, test_indexes, outliers_indexes
    
    # Generate all sets to test method, with train, test and outlier
    def generate_outliers(self, training, training_labels, test, test_labels, activity):
        
        training_indexes, test_indexes, outliers_indexes = \
        self.get_indexes(training_labels, test_labels, activity)
        return training.iloc[training_indexes,:], training_labels.iloc[training_indexes,:],\
                test.iloc[test_indexes,:], test_labels.iloc[test_indexes,:], \
                training.iloc[outliers_indexes,:], training_labels.iloc[outliers_indexes,:]
                
    def get_accuracy(self, classifier, training_outlier, test_outlier, outlier):
        classifier.fit(training_outlier)
        pred_outlier = classifier.predict(outlier)
        pred_test = classifier.predict(test_outlier)
        pred_training = classifier.predict(training_outlier)
        
        n_error_training = pred_training[pred_training == -1].size
        n_error_test = pred_test[pred_test == -1].size
        n_error_outlier = pred_outlier[pred_outlier == 1].size
        
        train_accuracy = (100 - (100 * (n_error_training / pred_training.size)))
        test_accuracy = (100 - (100 * (n_error_test / pred_test.size)))
        outliers_accuracy = (100 - (100 * (n_error_outlier / pred_outlier.size)))
        
        Debug.print_debug("train_accuracy = {}".format(train_accuracy))
        Debug.print_debug("test_accuracy = {}".format(test_accuracy))
        Debug.print_debug("outliers_accuracy = {}".format(outliers_accuracy))
        
        return train_accuracy, test_accuracy, outliers_accuracy
    
        
        
        
        