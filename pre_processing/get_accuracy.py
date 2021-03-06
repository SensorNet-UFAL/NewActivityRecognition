# -*- coding: utf-8 -*-
from models.model import Model
import pandas as pd
from collections import Counter
from outlier.outlier_commons import Outlier_Commons
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold
import statistics as st
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


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
            valid_data_from_each_person[p]["test"]["test_features"] = person_data['test']['test_features'].iloc[valid_indexes,:]
            valid_data_from_each_person[p]["test"]["test_labels"] = person_data['test']['test_labels'].iloc[valid_indexes,:]
            
            #clf.fit(valid_data_from_each_person[p]['training']['training_features'], valid_data_from_each_person[p]['training']['training_labels'])
            len_activity_before = len(person_data["test"]["test_labels"]["activity"].unique())
            len_activity_after = len(valid_data_from_each_person[p]["test"]["test_labels"]["activity"].unique())
            accuracies[p] = {"accuracy":clf.score(valid_data_from_each_person[p]['test']['test_features'], valid_data_from_each_person[p]['test']['test_labels']),"discarted":(len(pred)-len(valid_indexes))/len(pred), "discarted_activity":(len_activity_before-len_activity_after)}
            
        return accuracies
    
    
    ### COMEÇAR DAKI, TESTA NA FUNÇÃO
    def get_accuracy_cross(self, clf, relevant_features, y, n_to_split=5, threshold=0):
        accuracies = []
        precisions = []
        recalls = []
        f_scores = []
        discarteds = []
        len_activities = []
        spent_times = []
        skf = StratifiedKFold(n_splits=n_to_split, random_state=1, shuffle=False)
        for train_index, test_index in skf.split(relevant_features, y):
            
            x_train, x_test = relevant_features.loc[train_index], relevant_features.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
            train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
            
            x_test = x_test[test_valid_rows]
            y_test = y_test[test_valid_rows]
            x_train = x_train[train_valid_rows]
            y_train = y_train[train_valid_rows]
            
            #Getting unique labels to the data
            clf.classes_ = y_test.activity.unique()
            #Fit model
            clf.fit(x_train, y_train)
            #Getting the probability table
            pred = clf.predict_proba(x_test)
            pred = pd.DataFrame(pred, columns = clf.classes_)
            #Filtering data with probability greater than the threshold
            valid_indexes = self.get_indexes_with_valid_predictions(pred, threshold)
            new_x_test = x_test.iloc[valid_indexes,:]
            new_y_test = y_test.iloc[valid_indexes,:]
            start_time = time.time()
            pred = clf.predict(new_x_test)
            #accuracy = clf.score(new_x_test, new_y_test)
            end_time = time.time()
            spent_time = (end_time-start_time)/x_test.shape[0]
            #Calculating F-score
            accuracy = accuracy_score(new_y_test, pred)
            m = precision_recall_fscore_support(new_y_test, pred, average='weighted')
            
            accuracies.append(accuracy)
            precisions.append(m[0])
            recalls.append(m[1])
            f_scores.append(m[2])
            discarteds.append((len(pred)-len(valid_indexes))/len(pred))
            len_activities.append(len(new_y_test["activity"].unique()))
            spent_times.append(spent_time)
            
        return {"accuracy":st.mean(accuracies), "precision":st.mean(precisions), "recall":st.mean(recalls), "f-scores":st.mean(f_scores), "discarteds":st.mean(discarteds), "len_activity":st.mean(len_activities), "spent_time": st.mean(spent_times)}
    
    def simple_accuracy_with_valid_predictions(self, x_train, x_test, y_train, y_test, clf, threshold):
        #Getting unique labels to the data
        clf.classes_ = y_test.activity.unique()
        #Fit model
        clf.fit(x_train, y_train)
        #Getting the probability table
        pred = clf.predict_proba(x_test)
        pred = pd.DataFrame(pred, columns = clf.classes_)
        #Filtering data with probability greater than the threshold
        valid_indexes = self.get_indexes_with_valid_predictions(pred, threshold)
        new_x_test = x_test.iloc[valid_indexes,:]
        new_y_test = y_test.iloc[valid_indexes,:]
        start_time = time.time()
        pred = clf.predict(new_x_test)
        #accuracy = clf.score(new_x_test, new_y_test)
        end_time = time.time()
        spent_time = (end_time-start_time)/x_test.shape[0]
        
        #Calculating F-score
        accuracy = accuracy_score(new_y_test, pred)
        m = precision_recall_fscore_support(new_y_test, pred, average='weighted')
        
        accurary = {"accuracy":accuracy, "discarted":(len(pred)-len(valid_indexes))/len(pred), "len_activity":len(new_y_test["activity"].unique()), "spent_time":spent_time, "metrics":m}
        return accurary
        
    def plot_confusion_matrix(self, x_test, y_test, clf):
        class_names = y_test.activity.unique()
        disp = plot_confusion_matrix(clf, x_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize=None, xticks_rotation="vertical")
        disp.figure_.set_size_inches(18.5, 10.5, forward=True)
        print(disp.confusion_matrix)
        
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
                clf.classes_ = training_labels.activity.unique()
                clf.fit(training, training_labels)
                pred = clf.predict_proba(outlier)
                pred = pd.DataFrame(pred, columns = clf.classes_)
                index_pred = self.get_indexes_with_valid_predictions(pred, threshold)
                pred = clf.predict(training.iloc[index_pred, :])
                counter = Counter(pred)
                return_outlier = {}
                return_outlier["outlier activity"] = activity
                return_outlier["outlier pred"] = counter.most_common(1)[0][0]
                return return_outlier
            
    def stratified_kfold_accuracy_outlier(self, data, y, clf, threshold, person):
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
        outliers_commons = Outlier_Commons()
        return_accuracy = []
        for activity in y.activity.unique():
            accuracy_list_outlier = []
            accuracy_list_test = []
            for train_index, test_index in skf.split(data, y):
                x_train, x_test = data.iloc[train_index,:], data.iloc[test_index,:]
                y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
                test_valid_rows = np.isfinite(x_test[list(x_test.columns)[0]])
                train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
                
                x_test = x_test[test_valid_rows]
                y_test = y_test[test_valid_rows]
                x_train = x_train[train_valid_rows]
                y_train = y_train[train_valid_rows]
                
                training_aux, training_labels_aux, test_aux, test_labels_aux, outlier_aux, outlier_labels_aux = outliers_commons.generate_outliers(x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), activity)
                if training_aux.shape[0] > 0 and outlier_aux.shape[0] > 0:
                    clf.classes_ = training_labels_aux.activity.unique()
                    clf.fit(training_aux, training_labels_aux)
                    pred = clf.predict_proba(outlier_aux)
                    pred = pd.DataFrame(pred, columns = clf.classes_)
                    index_pred = self.get_indexes_with_valid_predictions(pred, threshold)
                    accuracy_list_outlier.append(1-(len(index_pred)/len(pred)))
                    
                    pred = clf.predict_proba(test_aux)
                    pred = pd.DataFrame(pred, columns = clf.classes_)
                    index_pred = self.get_indexes_with_valid_predictions(pred, threshold)
                    accuracy_list_test.append(len(index_pred)/len(pred))
            
            return_accuracy.append({"Person":person,"Activity Outlier": activity, "Accouracy Outlier":st.mean(accuracy_list_outlier), "Accouracy Teste":st.mean(accuracy_list_test)})
        return return_accuracy
                
            
            

    def get_outliers_confused_with_activities(self, data_from_each_person, clf, threshold):
        outliers_commons = Outlier_Commons()
        scaler = StandardScaler()
        for p in data_from_each_person:
                return_dataframe = pd.DataFrame(columns=["outlier_activity","outlier_pred"])
                person_data = data_from_each_person[p]
                training = pd.DataFrame(scaler.fit_transform(person_data['training']['training_features']))
                training_labels = person_data['training']['training_labels']
                test = pd.DataFrame(scaler.fit_transform(person_data['test']['test_features']))
                test_labels = person_data['test']['test_labels']
                return_accuracy = []
                for activity in training_labels.activity.unique():
                    training_aux, training_labels_aux, test_aux, test_labels_aux, outlier_aux, outlier_labels_aux = outliers_commons.generate_outliers(training.copy(), training_labels.copy(), test.copy(), test_labels.copy(), activity)
                    if training_aux.shape[0] > 0:
                        clf.classes_ = training_labels_aux.activity.unique()
                        clf.fit(training_aux, training_labels_aux)
                        pred = clf.predict_proba(outlier_aux)
                        pred = pd.DataFrame(pred, columns = clf.classes_)
                        index_pred = self.get_indexes_with_valid_predictions(pred, threshold)
                        return_accuracy.append({"Activity Outlier": activity, "Accouracy":(1-(len(index_pred)/len(pred)))})
                        pred = clf.predict(training_aux.iloc[index_pred, :])
                        counter = Counter(pred)
                        #print("outlier_activity: {} - outlier_pred: {}".format(activity, counter.most_common(1)[0][0]))
                        return_dataframe = pd.DataFrame([[activity, counter.most_common(1)[0][0]]], columns=["outlier_activity","outlier_pred"]).append(return_dataframe, ignore_index=True)
                        
                    else:
                        print("Empty Training: {} | Activity: {}".format(training_aux, activity))
                return return_dataframe, return_accuracy, data_from_each_person



