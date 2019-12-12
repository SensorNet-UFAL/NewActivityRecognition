# -*- coding: utf-8 -*-

# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.arcma_model import ARCMA_Model
from classifiers.base_classification import Base_Classification
import pandas as pd
from pre_processing.processing_db_files import Processing_DB_Files  
from utils.project import Project, slash
from scripts.save_workspace import save
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split

#===INITIALIZATION===#
Debug.DEBUG = 0
arcma = ARCMA_Model()
processing = Processing_DB_Files()
project = Project()
extra_trees = ExtraTreesClassifier(n_estimators = 1000, random_state=0)
base_classification = Base_Classification(arcma, extra_trees)
s = save()

p=1
relevant_features = s.load_var("umafall_relevant_features{}relevant_features_{}.pkl".format(slash, p))
y = s.load_var("umafall_relevant_features{}y_{}.pkl".format(slash, p))
y = pd.DataFrame(y, columns=[arcma.label_tag])




#def balanced_dataset(y, model,min_samples):
classes_counts = y["activity"].value_counts()
classes_counts.plot(kind='bar', title='Base ARCMA desbalanceada.')
threshould = 10
invalid_classes = []

#procurando valores abaixo do threshold
for index, value in classes_counts.iteritems():
    if value < threshould:
        invalid_classes.append(index)

#Retirando os valores abaixo do threshoud
invalid_indexes = []
for i in invalid_classes:
    invalid_indexes = invalid_indexes + list(y[y["activity"]==i].index)
y_new = y.copy()
y_new = y_new.drop(labels=invalid_indexes)
classes_counts = y_new["activity"].value_counts()
classes_counts.plot(kind='bar', title='Base ARCMA limiar 100.')

#Reamostragem para balancear
samples = list()
for classe, value in classes_counts.iteritems():
    samples.append(y_new[y_new["activity"]==classe].sample(classes_counts.min()))
y_new = pd.concat(samples)  
classes_counts = y_new["activity"].value_counts()
classes_counts.plot(kind='bar', title='Base ARCMA balanciada')

relevant_features_new = relevant_features.iloc[y_new.index-1,:]


'''skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=False)

for train_index, test_index in skf.split(relevant_features_new, y_new):
    x_train, x_test = relevant_features_new.loc[train_index], relevant_features_new.loc[test_index]
    y_train, y_test = y_new.loc[train_index], y_new.loc[test_index]
    
    test_valid_rows = np.isfinite(x_test[list(x_train.columns)[0]])
    train_valid_rows = np.isfinite(x_train[list(x_train.columns)[0]])
    x_test = x_test[test_valid_rows]
    y_test = y_test[test_valid_rows]
    x_train = x_train[train_valid_rows]
    y_train = y_train[train_valid_rows]
    
    extra_trees.fit(x_train, y_train)
    base_classification.get_accuracy.plot_confusion_matrix(x_test, y_test, extra_trees)
    break
'''
x_train, x_test, y_train, y_test = train_test_split(relevant_features_new, y_new, test_size=0.2, random_state=42)
extra_trees.fit(x_train, y_train)
base_classification.get_accuracy.plot_confusion_matrix(x_test, y_test, extra_trees)
print("Accuracy => {}".format(extra_trees.score(x_test, y_test)))