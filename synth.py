import numpy as np
import dataset

def gen(means,stds,n_size=100,ts_len=128,n_feats=12):
    params=[(mean_i,std_i) 
                for mean_i in means
                    for std_i in stds]
    n_cats=len(params)
    def make_sample(param_i):
        return np.random.normal(param_i[0],param_i[1],(ts_len,n_feats))
    samples=[]
    for i,param_i in enumerate(params):
        for j in range(n_size):
            for k in range(2):
                name_j="%d_%d_%d" % (i,k,j)
                data_j=make_sample(param_i)
                samples.append((name_j,data_j))	
    samples=dict(samples)
    return dataset.TSDataset(samples,'synth_gauss')    

gen([1,2],[1,2])
