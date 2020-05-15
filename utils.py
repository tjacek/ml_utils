import numpy as np
from sklearn.metrics import accuracy_score
import exper.cats,exper.inspect
import learn,feats,plot

def random_ensemble(in_path,n_cats=20,n=100):
    votes=feats.read_list(in_path)
    for i in range(2,n_cats):
        result=[subsample(votes,k=i) for j in range(n)]
        stats=np.amax(result),np.mean(result),np.median(result),np.amin(result)
        print("%s,%s,%s,%s" % stats)

def subsample(votes,k=5):
    samples=np.random.choice(votes,size=k,replace=True)
    result=exper.cats.simple_voting(samples)
    acc=accuracy_score(result[0],result[1])
    return acc

def absolute_errors(in_path):
    votes=feats.read_list(in_path)
    names=votes[0].info
    result=[get_result(vote_i)
        for vote_i in votes]
    result=np.array(result)
    result=np.sum(result,axis=0)
    print(np.mean(result))
    print(np.std(result))
    result= (result-np.mean(result))/np.std(result)
    pairs=[ (name_i,value_i) 
            for name_i,value_i in zip(names,result)
              if(value_i<-1)]
    print(list(zip(*pairs)))

def get_result(vote_i):
    y_true,y_pred=exper.inspect.pred(vote_i)
    return [ int(true_j==pred_j) 
                for true_j,pred_j in zip(y_true,y_pred)]

def ens_each(vote_path):
    votes=feats.read_list(vote_path)
    acc=[]
    for vote_i in votes:
        result_i=exper.inspect.pred(vote_i)
        acc_i=learn.each_acc(result_i)
        print(acc_i)
        acc.append(acc_i)
    acc=np.array(acc)
#    raise Exception(acc.shape)
    plot.heat_map(acc,["x","y"])

path_i="../result/ens5/LR/stats_sim"
path_i="../ens5/basic/LR"
ens_each(path_i)