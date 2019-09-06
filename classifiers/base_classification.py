# -*- coding: utf-8 -*-
from models.model import Model
from pre_processing.processing_db_files import Processing_DB_Files
from pre_processing.get_accuracy import Get_Accuracy
from statistics import mean
from utils.debug import Debug
   
class Base_Classification(object):
    def __init__(self, model:Model, clf):
        self.model = model
        self.get_accuracy = Get_Accuracy()
        self.processing = Processing_DB_Files()
        self.clf = clf
        
        super().__init__()
     
    #Finding the best window to improve the accuracy
    def find_best_window(self, interval):
        best_accuracy = {'window':1, 'accuracy':0}
        for w in interval:
            Debug.print_debug("WINDOWS SIZE: {}".format(w))
            data_all_people = self.model.load_training_data_from_all_people(w)
            features_all_people = self.processing.calculating_features_to_each_person(data_all_people, self.model)
            accuracies_dict = self.get_accuracy.simple_accuracy_mean_to_each_person(features_all_people, self.model, self.clf)
            accuracy_mean = mean(accuracies_dict.values())
            if accuracy_mean > best_accuracy["accuracy"]:
                best_accuracy["accuracy"] = accuracy_mean
                best_accuracy["window"] = w
        return best_accuracy
            
        
    def predict_for_all_people_with_proba(self, window, threshold):
        data_all_people = self.model.load_training_data_from_all_people(window)
        features_all_people = self.processing.calculating_features_to_each_person(data_all_people, self.model)
        accuracies_proba = self.get_accuracy.simple_accuracy_mean_to_each_person_with_proba(features_all_people, self.model, self.clf, threshold)
        accuracies = self.get_accuracy.simple_accuracy_mean_to_each_person(features_all_people, self.model, self.clf)
        return accuracies, accuracies_proba
    
    def predict_for_list_people_with_proba(self, window, list_people, threshold):
        data_list_people = self.model.load_training_data_from_list_people(window, list_people)
        features_all_people = self.processing.calculating_features_to_each_person(data_list_people, self.model)
        accuracies_proba = self.get_accuracy.simple_accuracy_mean_to_each_person_with_proba(features_all_people, self.model, self.clf, threshold)
        accuracies = self.get_accuracy.simple_accuracy_mean_to_each_person(features_all_people, self.model, self.clf)
        return accuracies, accuracies_proba
    
    def predict_outliers_for_list_people_with_proba(self, window, list_people, activity, threshold, remove_outliers = 0):
        data_list_people = self.model.load_training_data_from_list_people(window, list_people, remove_outliers)
        features_all_people = self.processing.calculating_features_to_each_person(data_list_people, self.model)
        #return self.get_accuracy.simple_accuracy_outlier_activity(features_all_people, self.model, self.clf, activity,threshold)
        return self.get_accuracy.get_outliers_confused_with_activities(features_all_people, self.model, self.clf,threshold)
        training, training_labels, test, test_labels, outlier, outlier_labels = self.get_accuracy.simple_accuracy_outlier_activity(features_all_people, self.model, self.clf, activity,threshold)
    
    