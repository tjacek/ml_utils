import numpy as np
import matplotlib.pyplot as plt
import exper,exper.cats,filtr,feats
import exper.inspect

#from exper.inspect import heat_map

def clf_selection(in_path):
#    corl_matrix=get_res_corelation(in_path)
#    mcc= np.mean(corl_matrix,axis=0)
    clf_ord=best_selection(in_path)
    print(clf_ord)
    return clf_ord

def best_selection(in_path):
    correct=exper.inspect.correct_votes(in_path,data="train")
    best=np.amax(correct,axis=0)
    for i,best_i in enumerate(best):
        correct[:,i][ correct[:,i] <best_i]=0.0
    correct[correct!=0]=1.0
    clf_best=np.sum(correct,axis=1)
    print(clf_best)
    return np.flip(np.argsort(clf_best))
#    raise Exception("OK")

def min_max_selection(in_path):
    correct=exper.inspect.correct_votes(in_path,data="train")
    print(correct)
    by_cat=np.sum(correct,axis=0) 
    worst_cat=correct[:,np.argmin(by_cat)]
    return np.flip(np.argsort(worst_cat))

def acc_selection(in_path):
    mcc=exper.inspect.clf_acc(in_path,data="train")
    return np.flip(np.argsort(mcc))

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

