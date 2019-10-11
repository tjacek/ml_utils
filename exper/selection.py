import numpy as np
import matplotlib.pyplot as plt
import exper,exper.cats,filtr,feats
import exper.inspect

from exper.inspect import heat_map

def clf_selection(in_path):
    datasets=[data_i.split()[0] for data_i in feats.read_list(in_path)]
    clf_corls=[ np.concatenate([dataset_corl(data_i,data_j) 
                     for data_i in datasets])
                        for data_j in datasets]
    n_clf=len(clf_corls)
    kl_matrix=[[np.corrcoef(clf_corls[i], clf_corls[j])[0][1]
                    for j in range(n_clf)]
                        for i in range(n_clf)]
    kl_matrix=np.array(kl_matrix)
    np.fill_diagonal(kl_matrix, 0)
    pairs=np.array(np.where(kl_matrix>0.7))
    return from_pairs(pairs)
#    exper.inspect.heat_map( (kl_matrix>0.7).astype(int) )

def dataset_corl(data1,data2):
    return [np.corrcoef(x1_i,x2_i)[0][1] 
                for x1_i,x2_i in zip(data1.X.T,data2.X.T)]

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

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#def clf_selection(in_path):
#    corl_train=exper.inspect.all_pred_corl(in_path,None,test=False)
#    corl_test=exper.inspect.all_pred_corl(in_path,None,test=True)
#    diff_corl= corl_test-corl_train
#    print(find_outliners(np.mean(diff_corl,axis=0),neg=True))    
#    exper.inspect.heat_map(diff_corl).show()  

#def __clf_selection(in_path):
#    corl_matrix=exper.inspect.all_pred_corl(in_path,None,test=True)
#    corl_matrix-= np.expand_dims(np.mean(corl_matrix,axis=0),axis=1)
#    corl_matrix/=np.expand_dims(np.std(corl_matrix,axis=0),axis=1)
    
#    better=(corl_matrix>1).astype(int)
#    clf_quality=np.sum(better,axis=1)
#    clf_quality= (clf_quality -np.mean(clf_quality))/np.std(clf_quality)   
#    print(clf_quality)
#    print(np.where(clf_quality>1))

#def _clf_selection(in_path):
#    corl_matrix=exper.inspect.all_pred_corl(in_path,None,test=False)
#    hard_cats=find_hard_cats(corl_matrix)
#    hard_matrix=corl_matrix[:,hard_cats]
#    hard_matrix-= np.expand_dims(np.mean(hard_matrix,axis=1),axis=1)
#    hard_matrix/=np.expand_dims(np.std(hard_matrix,axis=1),axis=1)
    
#    hard_matrix[hard_matrix>-1]=0
#    hard_matrix[hard_matrix<-1]=1

#    heat_map(hard_matrix).show()
#    clf_quality=np.sum(hard_matrix,axis=1)
#    print(clf_quality)
#    print(np.where(clf_quality>1))

def find_hard_cats(corl_matrix):
    cat_quality= np.mean(corl_matrix,axis=0)
    print(cat_quality)
    cat_quality-=np.mean(cat_quality)
    cat_quality/=np.std(cat_quality)
    return (cat_quality<-1)

def find_outliners(arr,neg=True):
    norm_arr= (arr-np.mean(arr))/np.std(arr)
    bool_arr= norm_arr<-1 if(neg) else norm_arr>1
    return np.where(bool_arr)