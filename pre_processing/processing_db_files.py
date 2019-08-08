# -*- coding: utf-8 -*-
from utils.debug import Debug
import random

class Processing_DB_Files(object):
    
    #Loading data with windows by people
    def load_training_data_by_window_by_people(self, dataset, filename, tablename, features, label,window_len, person_tag, person_column, additional_where = ""):
        list_raw_data = dataset.get_all_readings_from_person(filename, tablename, features, person_tag, person_column, additional_where)
        list_window = self.slice_by_window(list_raw_data, label, window_len)
        training, test = self.slice_to_training_test(list_window)
        return training, test
    
    
    def slice_by_window(self, dataframe, label, window_length):
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
    
    #Slices the set into training and testing
    def slice_to_training_test(self, dataset, training_proportion=0.8, seed=1):
    
        Debug.print_debug("Calculating the size of lists...")
        list_len = len(dataset)
        training_len = int((list_len*training_proportion))
        test_len = list_len - training_len
    
        #Initialzing the list of test and training
    
        Debug.print_debug("Initializing the sets of training and test...")
        training_list = dataset.copy()
        test_list = []
    
        #List with test indexes to put in test list
        Debug.print_debug("Calculating the random indexes...")
        random.seed(seed)
        test_index = random.sample(range(training_len), test_len)
    
        Debug.print_debug("Loop for create the sets of training e test...")
        for index in test_index:
            test_list.append(dataset[index])
            training_list.pop(index)
    
        Debug.print_debug("List len: {}".format(len(dataset)))
        Debug.print_debug("Training len: {}".format(len(training_list)))
        Debug.print_debug("Test len: {}".format(len(test_list)))
        Debug.print_debug("Training + test len: {}".format(len(training_list)+len(test_list)))
    
        return training_list, test_list
    
