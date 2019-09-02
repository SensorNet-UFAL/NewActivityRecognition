# -*- coding: utf-8 -*-
import numpy as np

class Outlier_Detection(object):
    
    def __init__(self):
        super().__init__()
        
    def get_indexes(self, training_labels, test_labels, label):
        
        training_indexes = np.where(training_labels != label)
        test_indexes = np.where(test_labels != label)
        outliers_indexes = np.where(training_labels == label)
    
        return training_indexes, test_indexes, outliers_indexes