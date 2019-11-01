import numpy as np
import random,itertools
from scipy.linalg import cholesky
from scipy.stats import norm,skewnorm
import dataset

def autoreg_gen(param_values,n_cats=20,n_size=500,ts_len=128,n_feats=12):
    def make_sample(param_i):
        c,std=param_i[-2:]
        params=np.array(param_i[:-2])
        lag=params.shape[0]
        noise=np.random.normal(0.0,std,(ts_len,))
        sample=[0.0 for i in range(lag)]
        for i in range(ts_len):
            x_i=sample[-lag:]
            sample.append(np.dot(x_i,params)+c + noise[i])
        return np.array(sample[lag:])
    return gen_template(param_values,make_sample,n_cats,n_size,n_feats,'synth_AR')

def skew_gen(means,stds,skew,n_cats=20,n_size=500,ts_len=128,n_feats=12):
    def make_sample(param_i):
        return skewnorm.rvs(a=param_i[2], loc=param_i[0], scale=param_i[1],size=ts_len)
    param_values=[means,stds,skew]
    return gen_template(param_values,make_sample,n_cats,n_size,n_feats,'synth_skew')

def normal_gen(means,stds,n_cats=20,n_size=500,ts_len=128,n_feats=12):
    def make_sample(param_i):
        return np.random.normal(param_i[0],param_i[1],(ts_len,))
    param_values=[means,stds]
    return gen_template(param_values,make_sample,n_cats,n_size,n_feats,'synth_gauss')

def gen_template(param_values,make_sample,n_cats=5,n_size=500,n_feats=12,name='synth'):
    params=list(itertools.product(*param_values))
    samples=[]
    for i in range(n_cats):
        param_subset=random.sample(params,n_feats)
        samples+=gen(i,param_subset,make_sample,n_size)
    return dataset.TSDataset(dict(samples),name)    

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
