import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
import dataset

def corl_samples(param_i):
    n_samples=400
    corl_i=param_i[0]
    r = np.array([[corl_i, 0.0],
                 [ 0.0, corl_i]])
    x = norm.rvs(size=(2, n_samples))
    c = cholesky(r, lower=True)
    y = np.dot(c, x)
    return y

def normal_gen(means,stds,n_size=100,ts_len=128,n_feats=12):
    params=[(mean_i,std_i) 
                for mean_i in means
                    for std_i in stds]
    n_cats=len(params)
    def make_sample(param_i):
        return np.random.normal(param_i[0],param_i[1],(ts_len,n_feats))
    samples=dict(gen(params,make_sample))
    return dataset.TSDataset(samples,'synth_gauss')    

def gen(params,make_sample):
    samples=[]
    for i,param_i in enumerate(params):
        for j in range(n_size):
            for k in range(2):
                name_j="%d_%d_%d" % (i,k,j)
                data_j=make_sample(param_i)
                samples.append((name_j,data_j)) 
    return make_sample

corl_samples([0.3])