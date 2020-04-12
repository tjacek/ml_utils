import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import exper,exper.cats,filtr,feats,learn
import exper.inspect,exper.curve

def acc_csv(in_path,out_path):        
    exper.curve.acc_to_csv(in_path,out_path,get_acc)

def make_plots(in_path,out_path):        
    exper.curve.all_curves(in_path,out_path,get_acc)

def get_acc(path_i,s_type="best"):
    if(type(s_type)==list):
        ord_i=s_type
    else:
        ord_i=clf_selection(path_i,s_type)
    results=selected_voting(path_i,ord_i)
    return [accuracy_score(result_i[0],result_i[1]) 
                for result_i in results]

def selection_result(path_i):
    clf_ord=clf_selection(path_i)
    results=selected_voting(path_i,clf_ord)
    acc_i=learn.acc_arr(results)
    k=np.argmax(acc_i)
    result_i=results[k]
    return result_i

def selected_voting(path_i,clf_ord):
    if(type(path_i)==str):
        data=feats.read_list(path_i) 
    else:
        data=path_i
    votes=[data[ord_ij] for ord_ij in clf_ord]
    n_clf=len(votes)
    results=[exper.cats.voting(votes[:(i+1)],None) 
                for i in range(n_clf)]
    return results

def clf_selection(in_path,s_type="best"):
    if(s_type=="best"):
        clf_ord= best_selection(in_path)
    if(s_type=="acc"):
        clf_ord=acc_selection(in_path)
    print(clf_ord)
    return clf_ord

def best_selection(in_path):
    correct=exper.inspect.correct_votes(in_path,data="train")
    best=np.amax(correct,axis=0)
    for i,best_i in enumerate(best):
        correct[:,i][ correct[:,i] <best_i]=0.0
    correct[correct!=0]=1.0
    print(correct)
    clf_best=np.sum(correct,axis=1)
    print(clf_best)
    return np.flip(np.argsort(clf_best))


def min_max_selection(in_path):
    correct=exper.inspect.correct_votes(in_path,data="train")
    print(correct)
    by_cat=np.sum(correct,axis=0) 
    worst_cat=correct[:,np.argmin(by_cat)]
    return np.flip(np.argsort(worst_cat))

def acc_selection(in_path):
    mcc=exper.inspect.clf_acc(in_path,data="train")
    return np.flip(np.argsort(mcc))

def mrc_selection(in_path):
    corl_matrix=get_res_corelation(in_path)
    mcc= np.mean(corl_matrix,axis=0)
    return np.argsort(mcc)

def get_res_corelation(in_path):
    datasets=[data_i.split()[0] for data_i in feats.read_list(in_path)]
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