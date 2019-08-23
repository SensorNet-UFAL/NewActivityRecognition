# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from pre_processing.processing_db_files import Processing_DB_Files
from models.hmp_model import HMP_Model





#===INITIALIZATION===#
Debug.DEBUG = 1
hmp = HMP_Model()
extra_trees = ExtraTreesClassifier(n_estimators = 1, max_depth=10, random_state=0)
#hmp.load_training_data_by_window_by_people("f1", 50)
#processing = Processing_DB_Files()
p = hmp.load_training_data_from_all_people()


#===PROCESSING===#
training, training_labels, test, test_labels = processing.calculating_features(hmp)
extra_trees.fit(training, training_labels)
pred_p = extra_trees.predict_proba(test)
pred_l_p = extra_trees.predict_log_proba(test)
print("Accuracy: {}".format(extra_trees.score(test, test_labels)))

#PLOT TREE IN FILE
'''
from sklearn.tree import export_graphviz
import pydot

str_tree = export_graphviz(extra_trees.estimators_[0], out_file="tree.dot", feature_names=training.columns, filled=True, special_characters=True, rotate=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')'''

