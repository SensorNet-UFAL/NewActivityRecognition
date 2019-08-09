# -*- coding: utf-8 -*-
from utils.debug import Debug
from models.hmp_model import HMP_Model

Debug.DEBUG = 1
hmp = HMP_Model()
hmp.load_training_data_by_window_by_people('f1', 50)


