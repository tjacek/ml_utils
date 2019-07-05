import numpy as np
import scipy.stats

def basic_stats(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return np.array([np.mean(feat_i),np.std(feat_i),
    	                scipy.stats.skew(feat_i),time_corl(feat_i)])

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)#1.0,step=step)
    return scipy.stats.pearsonr(x_i,feat_i)[0]