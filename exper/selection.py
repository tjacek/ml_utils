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

def cat_corls_matrix(datasets):
    clf_corls=[ np.concatenate([cat_corl(data_i,data_j) 
                     for data_i in datasets])
                        for data_j in datasets]
    n_clf=len(clf_corls)
    corl_matrix=[[np.corrcoef(clf_corls[i], clf_corls[j])[0][1]
                    for j in range(n_clf)]
                        for i in range(n_clf)]
    return np.array(corl_matrix)

def cat_corl(data1,data2):
    return [np.corrcoef(x1_i,x2_i)[0][1] 
                for x1_i,x2_i in zip(data1.X.T,data2.X.T)]

def find_outliners(samples,pos=True):
    norm_samples= (samples-np.mean(samples))/np.std(samples)
    indexes= (norm_samples<1.0) if(pos) else (norm_samples>1.0)
    return np.where(indexes)[0]

def from_pairs(pairs):
    values,counts = np.unique(pairs,return_counts=True)
    values=[values[k] for k in np.argsort(counts)]
    values.reverse()
    pairs=pairs.T
    right=[]
    for value_i in values:
        print(len(pairs))
        right.append(value_i)
        pairs=[pair_i for pair_i in pairs
                    if(np.all(pair_i!=value_i))]
        if(not len(pairs)):
            break
    wrong=set(values).difference(set(right))
    print(wrong)
    return wrong