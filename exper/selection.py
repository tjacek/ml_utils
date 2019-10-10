import numpy as np
import matplotlib.pyplot as plt
import exper,exper.cats,filtr,feats
import exper.inspect

from exper.inspect import heat_map

def clf_selection(in_path):
    corl_train=exper.inspect.all_pred_corl(in_path,None,test=False)
    corl_test=exper.inspect.all_pred_corl(in_path,None,test=True)
    diff_corl= corl_test-corl_train
    print(find_outliners(np.mean(diff_corl,axis=0),neg=True))    
#    corl_pred=np.abs(1.0-diff_corl)*corl_train
    exper.inspect.heat_map(diff_corl).show()  

def __clf_selection(in_path):
    corl_matrix=exper.inspect.all_pred_corl(in_path,None,test=True)
    corl_matrix-= np.expand_dims(np.mean(corl_matrix,axis=0),axis=1)
    corl_matrix/=np.expand_dims(np.std(corl_matrix,axis=0),axis=1)
    
    better=(corl_matrix>1).astype(int)

#    worse=(corl_matrix<-1).astype(int)
#    plt=exper.inspect.heat_map( better).show()    
#    print(np.sum(worse,axis=1))
    clf_quality=np.sum(better,axis=1)
    clf_quality= (clf_quality -np.mean(clf_quality))/np.std(clf_quality)   
    print(clf_quality)
    print(np.where(clf_quality>1))

def _clf_selection(in_path):
    corl_matrix=exper.inspect.all_pred_corl(in_path,None,test=False)
    hard_cats=find_hard_cats(corl_matrix)
    hard_matrix=corl_matrix[:,hard_cats]
    hard_matrix-= np.expand_dims(np.mean(hard_matrix,axis=1),axis=1)
    hard_matrix/=np.expand_dims(np.std(hard_matrix,axis=1),axis=1)
    
    hard_matrix[hard_matrix>-1]=0
    hard_matrix[hard_matrix<-1]=1

    heat_map(hard_matrix).show()
    clf_quality=np.sum(hard_matrix,axis=1)
    print(clf_quality)
    print(np.where(clf_quality>1))

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

#def plot_acc(acc):
#    print(acc)    
#    plt.plot(acc)
#    plt.ylabel("acc")
#    plt.show()