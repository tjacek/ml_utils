import numpy as np
from sklearn.cluster import KMeans
import feats,exper.cats,exper.inspect,learn
from exper.inspect import erorr_vector
from exper.ada_boost import weighted_voting
import ens,exper.curve
from sklearn.metrics import accuracy_score

def acc_csv(in_path,out_path):        
    exper.curve.acc_to_csv(in_path,out_path,get_acc)

def make_plots(in_path,out_path):        
    exper.curve.all_curves(in_path,out_path,get_acc)

def get_acc(path_i):
    votes=feats.read_list(path_i)
    best=[select_clf(path_i,k=i) for i in range(1,len(votes)+1)]
    return [ada_voting(best_i,acc=True) for best_i in best]

def select_clf(votes,k=4):
    if(type(votes)==str):
        votes=feats.read_list(votes)
    cls=clf_clust(votes,k)
    return [best_cls(cls_i) for cls_i in cls]

def ada_voting(best,acc=True):
    alphas=ada_weights(best)
    test=[best_i.split()[1] for best_i in best]
    result=weighted_voting(test,alphas)
    if(acc):
        return accuracy_score(result[1],result[0])	
    else:
        return learn.compute_score(result[1],result[0],as_str=True)

def clf_clust(votes,k=5):
    train=[ vote_i.split()[0] for vote_i in votes]
    error=[ exper.inspect.erorr_vector(train_i) 
            for train_i in train]
    kmeans=KMeans(k).fit(error)
    cls=[[] for i in range(k)]
    for i,cls_i  in enumerate(kmeans.labels_):
        cls[cls_i].append(votes[i])
    cls=[cls_i for cls_i in cls
           if(len(cls_i)>0)]
    return cls

def best_cls(cls_i):
    if(len(cls_i)==1):
        return cls_i[0]
    acc_i=[ np.mean(erorr_vector(cls_ij.split()[0])) 
            for cls_ij in cls_i]
    k=np.argmax(acc_i)
    return cls_i[k]

def ada_weights(data):
    train=[ data_i.split()[0] for data_i in data]
    K=train[0].n_cats()
    acc=np.array([ np.mean(vec_i) for vec_i in erorr_vector(train)])
    acc[acc>=1.0]=0.99
    return [ np.log((acc_i)/(1-acc_i)) +np.log(K-1) for acc_i in acc]

#def cls_to_csv(in_path,out_path,n_part=27):
#    def helper(path_i):
#        best=[select_clf(path_i,k=i) for i in range(2,n_part)]
#        acc=[ada_voting(best_i,acc=True) for best_i in best]
#        print(acc)        
#        k=np.argmax(acc)
#        stats=ada_voting(best[k],acc=False)
#        return "%d,%s"%(k+3,stats)
#    return ens.to_csv_template(in_path,out_path,helper)
