# -*- coding: utf-8 -*-
# IMPORTS #
from utils.debug import Debug
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from models.hmp_model import HMP_Model
from classifiers.base_classification import Base_Classification





#===INITIALIZATION===#
Debug.DEBUG = 0
hmp = HMP_Model()
#extra_trees = ExtraTreesClassifier(n_estimators = 10000, max_depth=1000, random_state=0) #Good performer
extra_trees = ExtraTreesClassifier(n_estimators = 100, max_depth=100, random_state=0) #To test
base_classification = Base_Classification(hmp, extra_trees)

#TEST CLASSIFICATION
#accuracies, accuracies_proba = base_classification.predict_for_list_people_with_proba(50, ["f1", "m1", "m2"] ,0.55)
#extra_trees_features_importance = extra_trees.feature_importances_


#TEST OUTLIER DETECTION

return_dataframe, return_accuracy, data_from_each_person = base_classification.predict_outliers_for_list_people_with_proba(50, ["f1", "m1", "m2"], "eat_soup" ,0.55, remove_outliers=0.05)

#===>> HISTORIC
#1º filtro de outliers implementado, mas não deu resultado.
#2º Utilizar MinMaxScaler para normalizar os dados.
#3º (Artigo -> https://scialert.net/fulltextmobile/?doi=jas.2010.950.958)Incluir os parametros do modelo ARIMA nas features - Pesquisar como utilizar o retorno de seasonal_decompose como entrada em algoritmo de classificação.


''' NOTES

Just persons with more than 12 distinct activities will are used:

"f1": 14;
"f2": 08;
"f3": 07;
"f4": 07;
"f5": 01;
"m1": 12;
"m10": 01;
"m11": 01;
"m2": 12;
"m3": 07;
"m4": 07;
"m5": 02;
"m6": 02;
"m7": 07;
"m8": 03;
"m9": 04;    

*Walk se confunde muito com climb_stairs

'''


