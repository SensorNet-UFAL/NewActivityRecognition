# -*- coding: utf-8 -*-
from utils.debug import Debug
from models.hmp_model import HMP_Model
from pre_processing.processing_db_files import Processing_DB_Files

Debug.DEBUG = 1
hmp = HMP_Model()
#hmp.load_training_data_by_window_by_people('f1', 50)
Processing_DB_Files.calculating_features(hmp)