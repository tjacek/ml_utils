import numpy as np
import matplotlib.pyplot as plt
import exper,exper.cats,filtr,feats
import exper.inspect

from exper.inspect import heat_map

def clf_selection(in_path):
    datasets=[data_i.split()[0] for data_i in feats.read_list(in_path)]
    datasets=[residuals(data_i) for data_i in datasets]
    corl_matrix=full_corls_matrix(datasets)
    mcc= np.mean(corl_matrix,axis=0)
    return np.argsort(mcc)#find_outliners(mcc,False)
    
def full_corls_matrix(datasets):
    clf_matrix=[[np.corrcoef(data_i.X.flatten(),data_j.X.flatten())[0][1]
                     for data_i in datasets]
                        for data_j in datasets]
    return np.array(clf_matrix)

def residuals( data_i):
    res_i=data_i.labels_array().astype(float)-data_i.X
    return feats.FeatureSet(res_i,data_i.info)
