# -*- coding: utf-8 -*-

class Processing_DB_Files:
    
    def load_training_data_with_window_from_person(dataset, filename, tablename, features, label,window_len, person_tag, person_column, additional_where = ""):
        list_raw_data = dataset.get_all_readings_from_person(filename, tablename, features, person_tag, person_column, additional_where)
        list_window = slice_by_window(list_raw_data, label, window_len)
        training, test = slice_to_training_test(list_window)
        return training, test