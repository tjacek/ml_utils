import sys
sys.path.append("..")
import numpy  as np
import diff_evol,ens

class Comb(object):
    def __init__(self,all_votes):
        self.corl=Corl(all_votes)
        self.mse=MSE(all_votes)
        self.iter=0

    def __call__(self,weights):
        self.iter+=1
        print(self.iter)
        return self.corl(weights)+self.mse(weights)     

class MSE(object):
    def __init__(self,all_votes):
        if(type(all_votes)==list):
            all_votes=ens.Votes(all_votes)
        self.all_votes=all_votes
        self.iter=0

    def __call__(self,weights):
        self.iter+=1
        print(self.iter)
        weights=weights/np.sum(weights)
        result=self.all_votes.weighted(weights)
        return mse_fun(result)

class Corl(object):
    def __init__(self,all_votes):
        if(type(all_votes)==list):
            all_votes=ens.Votes(all_votes)
        self.all_votes=all_votes	
        self.d=[ result_i.true_one_hot() 
                  for result_i in all_votes.results]
    
    def __call__(self,weights):
        weights=weights/np.sum(weights)
        results=self.all_votes.results
        C=corl(results,self.d)
        n_clf=len(self.all_votes)
        loss=0
        for i in range(n_clf):
            for j in range(n_clf):
                loss+=weights[i]*weights[j] * C[i,j] 	
        return 1.0*loss

def corl(results,d):
    n_clf=len(results)
    C=np.zeros((n_clf,n_clf))
    for i in range(n_clf):
        for j in range(n_clf):
            f_i=results[i].y_pred
            f_j=results[j].y_pred
            c_ij= (f_i-d)*(f_j-d)
            C[i,j]=np.mean(c_ij)
    return C

def mse_fun(result):
    y_true=result.true_one_hot()
    y_pred=result.y_pred
    squared_mean=[np.sum((true_i- pred_i)**2)
                for true_i,pred_i in zip(y_true,y_pred)]
    return  np.mean(squared_mean)

dir_path="../../3DHOI/"
binary_path="%s/ens/II/feats" % dir_path
base_path="%s/1D_CNN/feats" % dir_path
dtw_path="../../deep_dtw/dtw"
ae_path="../../best2/3_layers/feats"
common=[base_path,ae_path]
diff_voting=diff_evol.OptimizeWeights(Comb,
    maxiter=10,read=ens.read_multi)
result=diff_voting(common,binary_path)
result.report()
print(result.get_acc())