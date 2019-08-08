# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import os

class Model(object):
    
    file_path = ""
    table_name = ""
    features = ["x", "y", "z"]
    person_column = "person"
    
    
    def __init__(self, file, table_name):
        self.file_path = os.getcwd()+"\\data\\sql\\"+file
        self.table_name = table_name

    def get_all_readings_from_person(self, person_tag, additional_where = ""):
        print(self.file_path)
        dataset = sqlite3.connect(self.file_path)
        if len(additional_where) > 0:
            to_return = self.get_data_sql_query("select {} from {} where {} = {} {}".format(', '.join(self.features), self.table_name, self.person_column, person_tag, additional_where), dataset)
        else:
            to_return = self.get_data_sql_query("select {} from {} where {} = '{}'".format(', '.join(self.features), self.table_name, self.person_column, person_tag), dataset)
        return to_return

    def get_data_sql_query(self, query,dataset):
        return pd.read_sql_query(query, dataset)    

