import numpy as np
import random
from scipy.linalg import cholesky
from scipy.stats import norm
import dataset

def normal_gen(means,stds,n_cats=5,n_size=500,ts_len=128,n_feats=12):
    params=[(mean_i,std_i) 
                for mean_i in means
                    for std_i in stds]
    if(len(params)>n_feats):
        params=params[:n_feats]
    def make_sample(param_i):
        return np.random.normal(param_i[0],param_i[1],(ts_len,))
    samples=[]
    for i in range(n_cats):
        random.shuffle(params)
        samples+=gen(i,params,make_sample,n_size)
    return dataset.TSDataset(dict(samples),'synth_gauss')    

def gen(cat_i,params,make_sample,n_size):
    samples=[]
    for j in range(n_size):
        seq_j=np.array([make_sample(param_i) for param_i in params]).T
        name_j="%d_%d_%d" % (cat_i,(j%2),j)
        samples.append((name_j,seq_j))
    return samples

def corl_samples(param_i):
    n_samples=400
    corl_i=param_i[0]
    r = np.array([[corl_i, 0.0],
                 [ 0.0, corl_i]])
    x = norm.rvs(size=(2, n_samples))
    c = cholesky(r, lower=True)
    y = np.dot(c, x)
    return y
