# -*- coding: utf-8 -*-
from models.umafall_model import UMAFALL_Model
import sqlite3
import pandas as pd


umafall = UMAFALL_Model()
dataset = sqlite3.connect(umafall.file_path)

#RIGHTPOCKET
data_rightpocket_accelerometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(0,0), dataset) # RIGHTPOCKET and ACCELEROMETER
data_rightpocket_gyroscope = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(0,1), dataset) # RIGHTPOCKET and GYROSCOPE
data_rightpocket_magnetometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(0,2), dataset) # RIGHTPOCKET and MAGNETOMETER

#CHEST
data_chest_accelerometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(1,0), dataset) # CHEST and ACCELEROMETER
data_chest_gyroscope = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(1,1), dataset) # CHEST and GYROSCOPE
data_chest_magnetometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(1,2), dataset) # CHEST and MAGNETOMETER



#WRIST
data_wrist_accelerometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(3,0), dataset) # WRIST and ACCELEROMETER
data_wrist_gyroscope = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(3,1), dataset) # WRIST and GYROSCOPE
data_wrist_magnetometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(3,2), dataset) # WRIST and MAGNETOMETER


#ANKLE
data_ankle_accelerometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(4,0), dataset) # ANKLE and ACCELEROMETER
data_ankle_gyroscope = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(4,1), dataset) # ANKLE and GYROSCOPE
data_ankle_magnetometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(4,2), dataset) # ANKLE and MAGNETOMETER

#WAIST
data_waist_accelerometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(2,0), dataset) # WAIST and ACCELEROMETER
data_waist_gyroscope = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(2,1), dataset) # WAIST and GYROSCOPE
data_waist_magnetometer = pd.read_sql_query("select * from umafall where body = '{}' and sensor = '{}' and person=1".format(2,2), dataset) # WAIST and MAGNETOMETER

