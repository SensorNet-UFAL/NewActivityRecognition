# -*- coding: utf-8 -*-
from models.arcma_model import ARCMA_Model
from models.hmp_model import HMP_Model
from models.umafall_model import UMAFALL_Model


#ARCMA
arcma = ARCMA_Model()
arcma.load_training_data_by_window_by_people(1,36)
sample_arcma = arcma.training[0]
sample_arcma_to_plot = sample_arcma.drop("activity",1)
sample_arcma_to_plot.plot(subplots=True, legend=True, figsize=(20,10))

#HMP
hmp = HMP_Model()
hmp.load_training_data_by_window_by_people("f1",36)
sample_hmp = hmp.training[0]
sample_hmp_to_plot = sample_hmp.drop("activity",1)
sample_hmp_to_plot.plot(subplots=True, legend=True, figsize=(20,10))

#UMAFALL
umafall = UMAFALL_Model()
umafall.load_training_data_by_window_by_people("1",10, additional_where="and sensor=2 and body=3")
sample_umafall = umafall.training[0]
sample_umafall_to_plot = sample_umafall.drop("activity",1)
sample_umafall_to_plot.plot(subplots=True, legend=True, figsize=(20,10))
