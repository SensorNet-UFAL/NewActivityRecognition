# -*- coding: utf-8 -*-
from utils.debug import Debug
from models.hmp_model import HMP_Model
from pre_processing.processing_db_files import Processing_DB_Files
from outlier.outlier_commons import Outlier_Commons
import pandas as pd

import matplotlib.pyplot as plt

# INITIALIZATION #
Debug.DEBUG = 1
hmp = HMP_Model()
outlier = Outlier_Commons()

# PROCESSING #
processing = Processing_DB_Files()
hmp.load_training_data_by_window_by_people('f1', 50)
training, training_labels, test, test_labels = processing.calculating_features(hmp)

p = pd.DataFrame(training)
box = p.boxplot(column=[1])
box["medians"]
