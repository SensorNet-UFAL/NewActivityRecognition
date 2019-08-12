# -*- coding: utf-8 -*-
#from utils.debug import Debug
from models.model import Model

class Processing_DB_Files(object):
    
    def __init__():
        super().__init__()
        
    def calculating_features(model:Model):
        
        label = model.label_tag
        x_label= model.features[0]
        y_label=model.features[1]
        z_label=model.features[2]
        
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
            if len(np.unique(d[label_name])) < 2:
                x = d.loc[:, x_label]
                y = d.loc[:, y_label]
                z = d.loc[:, z_label]
    
                #convert itens from pandas series to floats
                x = x.astype('float32')
                y = y.astype('float32')
                z = z.astype('float32')
                # INTEGRATION #
                ax = round(get_integration(normalization(x)), 3)
                ay = round(get_integration(normalization(y)), 3)
                az = round(get_integration(normalization(z)), 3)
                #i = round(get_integration(d.loc[:, "x"]), 3)
                x_integration.append(ax)
                y_integration.append(ay)
                z_integration.append(az)
    
                # RMS #
                x_r = round(get_rms(x),3)
                y_r = round(get_rms(x), 3)
                z_r = round(get_rms(x), 3)
    
                x_rms.append(x_r)
                y_rms.append(y_r)
                z_rms.append(z_r)
    
                # MINMAX #
                x_mm = round(get_minmax(x), 3)
                y_mm = round(get_minmax(y), 3)
                z_mm = round(get_minmax(z), 3)
    
                x_minmax.append(x_mm)
                y_minmax.append(y_mm)
                z_minmax.append(z_mm)
    
                # MEAN #
                x_m = round(get_mean(x), 3)
                y_m = round(get_mean(y), 3)
                z_m = round(get_mean(z), 3)
    
                x_mean.append(x_m)
                y_mean.append(y_m)
                z_mean.append(z_m)
    
                # STANDARD DESVIATION #
                x_sd = round(get_std(x), 3)
                y_sd = round(get_std(y), 3)
                z_sd = round(get_std(z), 3)
    
                x_std.append(x_sd)
                y_std.append(y_sd)
                z_std.append(z_sd)
    
                # KURTOSIS #
                x_k = round(get_kurtosis(x), 3)
                y_k = round(get_kurtosis(y), 3)
                z_k = round(get_kurtosis(z), 3)
    
                x_kurtosis.append(x_k)
                y_kurtosis.append(y_k)
                z_kurtosis.append(z_k)
    
                # SKWESS #
                x_sk = round(get_skew(x), 3)
                y_sk = round(get_skew(y), 3)
                z_sk = round(get_skew(z), 3)
    
                x_skew.append(x_sk)
                y_skew.append(y_sk)
                z_skew.append(z_sk)
    
                # CORRELATION #
                x_y.append(round(get_correlation(x, y), 3))
                x_z.append(round(get_correlation(x, z), 3))
                y_z.append(round(get_correlation(y, z), 3))
    
                # GET LABEL #
                label = d[label_name].iloc[0]
                label_list.append(label) #Get label for d
            else:
                discart = discart + 1
    
        print("Total of discarded windows: {}".format(discart))
        #Initializing features array
        features = create_array_features([x_integration, y_integration, z_integration, x_rms, y_rms, z_rms,
                                          x_minmax, y_minmax, z_minmax, x_mean, y_mean, z_mean,
                                          x_std, y_std, z_std, x_kurtosis, y_kurtosis, z_kurtosis,
                                          x_y, x_z, y_z])
        print("Features Shape: {}".format(features.shape))
        #Initializing labels array
        labels = np.array(label_list)
        #labels = np.reshape(labels, (labels.shape[0],n1
        # 1))
        return features, labels
            
        
