# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:44:01 2020

@author: wylken.machado
"""

import sqlite3
import pandas as pd
from models.arcma_model import ARCMA_Model
from models.hmp_model import HMP_Model
from models.umafall_model import UMAFALL_Model


#ARCMA

arcma = ARCMA_Model()
dataset = sqlite3.connect(arcma.file_path)
data = pd.read_sql_query("select * from arcma", dataset)

# Verificar Valor Nulo
bool_series = pd.isnull(data["activity"])
data[bool_series]


data["activity"].value_counts()
data["person"].value_counts()


#HMP

hmp = HMP_Model()
dataset = sqlite3.connect(hmp.file_path)
data = pd.read_sql_query("select * from hmp", dataset)

# Verificar Valor Nulo
bool_series = pd.isnull(data["z"])
data[bool_series]


data["activity"].value_counts()
data["person"].value_counts()


#UMAFALL

umafall = UMAFALL_Model()
dataset = sqlite3.connect(umafall.file_path)
data = pd.read_sql_query("select * from umafall", dataset)

# Verificar Valor Nulo
bool_series = pd.isnull(data["person"])
data[bool_series]


data["activity"].value_counts()
data["person"].value_counts()
