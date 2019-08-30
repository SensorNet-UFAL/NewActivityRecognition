# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification
import pandas as pd





#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
extra_trees = ExtraTreesClassifier(n_estimators = 100, max_depth=100, random_state=0)
base_classification = Base_Classification(hmp, extra_trees)
proba, pred, dataset = base_classification.predict_with_proba(50, 50)
pred = pd.DataFrame(pred, columns=["Activity"])
c = dataset["test"]["test_labels"].activity.unique()
r = pd.DataFrame(proba, columns = extra_trees.classes_)
#best_accuracy = base_classification.find_best_window(range(10, 101, 50))  



#PLOT TREE IN FILE
'''
from sklearn.tree import export_graphviz
import pydot

str_tree = export_graphviz(extra_trees.estimators_[0], out_file="tree.dot", feature_names=training.columns, filled=True, special_characters=True, rotate=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')'''

