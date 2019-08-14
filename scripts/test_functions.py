# -*- coding: utf-8 -*-
from utils.debug import Debug
from models.hmp_model import HMP_Model
from pre_processing.processing_db_files import Processing_DB_Files
from outlier.outlier_detection import Outlier_Detection

# INITIALIZATION #
Debug.DEBUG = 1
hmp = HMP_Model()
outlier = Outlier_Detection()

# PROCESSING #
processing = Processing_DB_Files()
hmp.load_training_data_by_window_by_people('f1', 50)
training, training_labels, test, test_labels = processing.calculating_features(hmp)
training_index, test_index, outliers_index = outlier.get_indexes(training_labels, test_labels, "drink_glass")