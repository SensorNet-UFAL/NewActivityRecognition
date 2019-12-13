# -*- coding: utf-8 -*-
from utils.debug import Debug
from utils.project import Project, slash
import sqlite3
import pandas as pd
import numpy as np
import random
from pyod.models.knn import KNN

class Model(object):
    
    file_path = ""
    table_name = ""
    person_column = "person"
    features = ["x", "y", "z", "activity"]
    label_tag = "activity"
    person_column = "person"
    training, test, data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data_with_window = None    
    
    def __init__(self, file, table_name):
        self.file_path = Project.project_root+"{}data{}sql{}".format(slash, slash, slash)+file
        self.table_name = table_name

    def get_all_readings_from_person(self, person_tag, remove_outliers = 0, additional_where = ""):
        #Debug.print_debug(self.file_path)
        print(self.file_path)
        dataset = sqlite3.connect(self.file_path)
        if len(additional_where) > 0:
            to_return = self.get_data_sql_query("select {} from {} where {} like {} {}".format(', '.join(self.features), self.table_name, self.person_column, person_tag, additional_where), dataset)
        else:
            to_return = self.get_data_sql_query("select {} from {} where {} like '{}'".format(', '.join(self.features), self.table_name, self.person_column, person_tag), dataset)
        self.data = to_return
        if(remove_outliers > 0):
            knn =  KNN(contamination=remove_outliers)
            to_return_aux = to_return.copy()
            to_return_aux = to_return_aux.drop(self.label_tag,1)
            knn.fit(to_return_aux)
            pred = knn.predict(to_return_aux)
            to_return = to_return.iloc[np.where(pred == 0)[0], :]
            
        return to_return
    
    def get_labels_to_person(self, person):
        dataset = sqlite3.connect(self.file_path)
        to_return = self.get_data_sql_query("SELECT DISTINCT {} from {} where person like '{}'".format(self.label_tag, self.table_name, person), dataset)
        return to_return[self.label_tag]

    def get_data_sql_query(self, query,dataset):
        return pd.read_sql_query(query, dataset)    
    
    #Loading data with windows by people
    def load_training_data_by_window_by_people(self, person_tag, window_len, training_proportion=0.8, seed=1, additional_where = ""):
        list_raw_data = self.get_all_readings_from_person(person_tag, additional_where = additional_where)
        list_window = self.slice_by_window(list_raw_data, window_len)
        self.training, self.test = self.slice_to_training_test(list_window, training_proportion, seed)
    #Loading data with windows by people
    def load_training_data_by_people(self, person_tag, remove_outliers = 0, additional_where = ""):
        list_raw_data = self.get_all_readings_from_person(person_tag, remove_outliers, additional_where = additional_where)
        return list_raw_data
    
    #Loading data from list of people
    def load_training_data_from_list_people(self, window_len, list_people, remove_outliers = 0, training_proportion=0.8, seed=1, additional_where = ""):
        list_of_peoples_data = {}
        for p in list_people:
            aux = self.slice_by_window(self.load_training_data_by_people(p, remove_outliers, additional_where = additional_where), window_len)
            training_test = {}
            training_test['training'], training_test['test'] = self.slice_to_training_test(aux, training_proportion, seed)
            list_of_peoples_data[p] = training_test
        self.data_with_window = list_of_peoples_data
        return list_of_peoples_data
    def load_data_to_tsfresh(self, window_len, people, remove_outliers = 0, additional_where = ""):
        aux = self.load_training_data_by_people(people, remove_outliers, additional_where = additional_where)
        return self.slice_by_window_tsfresh(aux)
    #Loading data from all people
    def load_training_data_from_all_people(self, window_len, training_proportion=0.8, seed=1):
        dataset = sqlite3.connect(self.file_path)
        peoples = self.get_data_sql_query("select distinct {} from {}".format(self.person_column, self.table_name), dataset)
        list_of_peoples_data = {}
        for p in peoples[self.person_column]:
            aux = self.slice_by_window(self.load_training_data_by_people(p), window_len)
            training_test = {}
            training_test['training'], training_test['test'] = self.slice_to_training_test(aux, training_proportion, seed)
            list_of_peoples_data[p] = training_test
            
        return list_of_peoples_data
    
    def slice_by_window(self, dataframe, window_length):
        index = 0
        dataframe_len = len(dataframe)
        result = []
        while(index < dataframe_len):
            try:
                l = dataframe.iloc[index:(index+window_length)]
                result.append(l)
                index = index + window_length
            except Exception as e:
                Debug.print_debug(e)
                break
        return result
    def slice_by_window_tsfresh(self, dataframe, window_length):
        index = 0
        dataframe_len = len(dataframe)
        result = []
        window_id = 1
        y = list()
        while(index < dataframe_len):
            try:
                #print("Step {}".format(window_id))
                l = dataframe.iloc[index:(index+window_length)]
                l["id"] = window_id
                l["time"] = pd.Series(range(index, (index+window_length)), index=l.index)
                y.append(l["activity"].value_counts(ascending=False).idxmax())
                result.append(l)
                index = index + window_length
                window_id +=1
            except Exception as e:
                Debug.print_debug(e)
                break
        return pd.concat(result), pd.Series(y)
    #Slices the set into training and testing
    def slice_to_training_test(self, dataset, training_proportion=0.8, seed=1):
    
        Debug.print_debug("Calculating the size of lists...")
        list_len = len(dataset)
        training_len = int((list_len*training_proportion))
        test_len = list_len - training_len
    
        #Initialzing the list of test and training
    
        Debug.print_debug("Initializing the sets of training and test...")
        training_list_aux = dataset.copy()
        test_list = []
    
        #List with test indexes to put in test list
        Debug.print_debug("Calculating the random indexes...")
        random.seed(seed)
        test_index = random.sample(range(len(dataset)), test_len)
        #test_index = random.sample(range(training_len), test_len)
    
        Debug.print_debug("Loop for create the sets of training e test...")
        for index in test_index:            
            test_list.append(dataset[index])
            training_list_aux[index] = pd.DataFrame()
        
        training_list = list(filter(lambda a: not a.empty, training_list_aux))
        Debug.print_debug("List len: {}".format(len(dataset)))
        Debug.print_debug("Training len: {}".format(len(training_list)))
        Debug.print_debug("Test len: {}".format(len(test_list)))
        Debug.print_debug("Training + test len: {}".format(len(training_list)+len(test_list)))
    
        return training_list, test_list

