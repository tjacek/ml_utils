import numpy as np
import matplotlib.pyplot as plt
import exper,exper.cats,filtr,feats
import exper.inspect

#from exper.inspect import heat_map

def clf_selection(in_path):
    corl_matrix=get_res_corelation(in_path)
    mcc= np.mean(corl_matrix,axis=0)
    print(mcc)
    return np.argsort(mcc)#find_outliners(mcc,False)

def get_res_corelation(in_path):
    datasets=[data_i.split()[1] for data_i in feats.read_list(in_path)]
    datasets=[errors(data_i) for data_i in datasets]
    return full_corls_matrix(datasets)
    
def full_corls_matrix(datasets):
    clf_matrix=[[np.corrcoef(data_i.X.flatten(),data_j.X.flatten())[0][1]
                     for data_i in datasets]
                        for data_j in datasets]
    return np.array(clf_matrix)

def errors(data_i):
    y_true=data_i.get_labels()
    y_pred=[np.argmax(x_i) for x_i in data_i.X] 
    error=[ float(true_j!=pred_j) 
                for true_j,pred_j in zip(y_true,y_pred)]
    return feats.FeatureSet(np.array(error),data_i.info)

def residuals( data_i):
    res_i=np.abs(data_i.labels_array().astype(float)-data_i.X)
    return feats.FeatureSet(res_i,data_i.info)

