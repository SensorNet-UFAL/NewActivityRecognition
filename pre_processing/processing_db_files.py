# -*- coding: utf-8 -*-
#from utils.debug import Debug
from models.model import Model
import numpy as np
import math
from numpy import trapz
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats.stats import pearsonr

class Processing_DB_Files(object):
    
    def __init__(self):
        super().__init__()
        
    def calculating_features(self, model:Model):

        training, training_labels = self.calculating_features_raw(model.training, model.label_tag, model.features[0], model.features[1], model.features[2])
        test, test_labels = self.calculating_features_raw(model.test, model.label_tag, model.features[0], model.features[1], model.features[2])
        return training, training_labels, test, test_labels
        
    def normalization(self, dataset):
        dataset_n = []
        mx = max(dataset)
        mn = min(dataset)
        for i in dataset:
            if(mx - mn == 0):
                return dataset
            dataset_n.append(round(((i-mn)/(mx-mn)), 3))
        return dataset_n
        
    def get_rms(self, dataset):
        x_2 = np.power(dataset, 2)
        return math.sqrt(np.sum(x_2)/len(dataset))

    def get_mean(self, dataset):
        return np.mean(dataset)
    
    def get_std(self, dataset):
        return np.std(dataset)
    
    def get_minmax(self, dataset):
        return max(dataset) - min(dataset)
    
    def get_integration(self, dataset):
        return trapz(dataset)
    
    def get_kurtosis(self, dataset):
        return kurtosis(dataset)
    
    def get_skew(self, dataset):
        return skew(dataset)
    
    def get_correlation(self, var1, var2):
        c = pearsonr(var1, var2)[0]
        if np.isnan(c):
            c = 0.0
        return c
    
    def create_array_features(self, features_data_list):
        features = np.zeros(shape=(len(features_data_list[0]), len(features_data_list)))
        features = features.astype('float32')
        for index, features_data in enumerate(features_data_list):
            features[:, index] = features_data
        return features
        
    def calculating_features_raw(self, dataframe_list, label_tag, x_label, y_label, z_label):
        
        # AUX #
        label_list = []
        discart = 0
        
        # INTEGRATION #
        x_integration, y_integration, z_integration = [], [], []
    
        # RMS #
        x_rms, y_rms, z_rms = [], [], []
    
        # MINMAX #
        x_minmax, y_minmax, z_minmax = [], [], []
    
        # MEAN #
        x_mean, y_mean, z_mean = [], [], []
    
        # STANDARD DESVIATION #
        x_std, y_std, z_std = [], [], []
    
        # KURTOSIS #
        x_kurtosis, y_kurtosis, z_kurtosis = [], [], []
    
        # SKWESS #
        x_skew, y_skew, z_skew = [], [], []
    
        # CORRELATION #
        x_y, x_z, y_z = [], [], []
    
        for d in dataframe_list:
            #print(d[label_name])
            if len(np.unique(d[label_tag])) < 2:
                x = d.loc[:, x_label]
                y = d.loc[:, y_label]
                z = d.loc[:, z_label]
    
                #convert itens from pandas series to floats
                x = x.astype('float32')
                y = y.astype('float32')
                z = z.astype('float32')
                # INTEGRATION #
                ax = round(self.get_integration(self.normalization(x)), 3)
                ay = round(self.get_integration(self.normalization(y)), 3)
                az = round(self.get_integration(self.normalization(z)), 3)
                #i = round(get_integration(d.loc[:, "x"]), 3)
                x_integration.append(ax)
                y_integration.append(ay)
                z_integration.append(az)
    
                # RMS #
                x_r = round(self.get_rms(x),3)
                y_r = round(self.get_rms(x), 3)
                z_r = round(self.get_rms(x), 3)
    
                x_rms.append(x_r)
                y_rms.append(y_r)
                z_rms.append(z_r)
    
                # MINMAX #
                x_mm = round(self.get_minmax(x), 3)
                y_mm = round(self.get_minmax(y), 3)
                z_mm = round(self.get_minmax(z), 3)
    
                x_minmax.append(x_mm)
                y_minmax.append(y_mm)
                z_minmax.append(z_mm)
    
                # MEAN #
                x_m = round(self.get_mean(x), 3)
                y_m = round(self.get_mean(y), 3)
                z_m = round(self.get_mean(z), 3)
    
                x_mean.append(x_m)
                y_mean.append(y_m)
                z_mean.append(z_m)
    
                # STANDARD DESVIATION #
                x_sd = round(self.get_std(x), 3)
                y_sd = round(self.get_std(y), 3)
                z_sd = round(self.get_std(z), 3)
    
                x_std.append(x_sd)
                y_std.append(y_sd)
                z_std.append(z_sd)
    
                # KURTOSIS #
                x_k = round(self.get_kurtosis(x), 3)
                y_k = round(self.get_kurtosis(y), 3)
                z_k = round(self.get_kurtosis(z), 3)
    
                x_kurtosis.append(x_k)
                y_kurtosis.append(y_k)
                z_kurtosis.append(z_k)
    
                # SKWESS #
                x_sk = round(self.get_skew(x), 3)
                y_sk = round(self.get_skew(y), 3)
                z_sk = round(self.get_skew(z), 3)
    
                x_skew.append(x_sk)
                y_skew.append(y_sk)
                z_skew.append(z_sk)
    
                # CORRELATION #
                x_y.append(round(self.get_correlation(x, y), 3))
                x_z.append(round(self.get_correlation(x, z), 3))
                y_z.append(round(self.get_correlation(y, z), 3))
    
                # GET LABEL #
                label = d[label_tag].iloc[0][0]
                label_list.append(label) #Get label for d
            else:
                discart = discart + 1
    
        print("Total of discarded windows: {}".format(discart))
        #Initializing features array
        features = self.create_array_features([x_integration, y_integration, z_integration, x_rms, y_rms, z_rms,
                                          x_minmax, y_minmax, z_minmax, x_mean, y_mean, z_mean,
                                          x_std, y_std, z_std, x_kurtosis, y_kurtosis, z_kurtosis,
                                          x_y, x_z, y_z])
        print("Features Shape: {}".format(features.shape))
        #Initializing labels array
        labels = np.array(label_list)
        #labels = np.reshape(labels, (labels.shape[0],n1
        # 1))
        return features, labels
            
        
