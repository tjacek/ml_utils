import numpy as np
import pickle
import unify

class DTWPairs(object):
    def __init__(self,pairs):
        self.pairs=pairs
    
    def save(self,out_path):
    	with open(out_path, 'wb') as handle:
            pickle.dump(self.pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read(in_path):
    with open(in_path, 'rb') as handle:
        return pickle.load(handle)

def make_pairwise_distance(ts_dataset):
    names=ts_dataset.ts_names()
    n_ts=len(names)   
    pairs_dict={ name_i:{name_i:0.0}
                    for name_i in names}
    for i in range(1,n_ts):
        print(i)
        for j in range(0,i):
            name_i,name_j=names[i],names[j]
            distance_ij=dtw_metric(ts_dataset[name_i],ts_dataset[name_j])
            pairs_dict[name_i][name_j]=distance_ij
            pairs_dict[name_j][name_i]=distance_ij
    return DTWPairs(pairs_dict)

def dtw_metric(s,t):
    dtw,n,m=prepare_matrix(s,t)
    for i in range(1,n+1):
        for j in range(1,m+1):
            t_i,t_j=s[i-1],t[j-1]
            diff=t_i-t_j
            cost= np.dot(diff,diff)          
            dtw[i][j]=cost+min([dtw[i-1][j],dtw[i][j-1],dtw[i-1][j-1]])
    return np.sqrt(dtw[n][m])

def prepare_matrix(s,t):
    n=len(s)
    m=len(t)
    cost_matrix=np.zeros((n+1,m+1),dtype=float)
    for i in range(1,n+1):
        cost_matrix[i][0]=np.inf
    for i in range(1,m+1):
        cost_matrix[0][i]=np.inf
    return cost_matrix,n,m

if __name__ == "__main__":
    ts_dataset=unify.read("mra/max_z")
    make_pairwise_distance(ts_dataset).save('dtw_pairs')