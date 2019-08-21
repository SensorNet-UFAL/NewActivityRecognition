# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from models.hmp_model import HMP_Model
from outlier.outlier_commons import Outlier_Commons
from sklearn import svm

#===INITIALIZATION===#
Debug.DEBUG = 1
outlier_commons = Outlier_Commons()

#===PROCESSING===#
# Getting sets with training, test and outlier data
training_outlier, training_labels_outlier, test_outlier, test_labels_outlier, \
outlier, outlier_labels = outlier_commons.outlier_prepare(HMP_Model(), "f1", 50, "drink_glass")

classifier = svm.OneClassSVM(gamma='auto')

outlier_commons.get_accuracy(classifier, training_outlier[:,18:21], test_outlier[:,18:21], outlier[:,18:21])
