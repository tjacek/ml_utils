import numpy as np
import scipy.stats
import smooth

class Nonlinearity(object):
    def __init__(self,k=2,smoothing=None,epsilion=0.01):
        if(not smoothing):
            smoothing=smooth.Fourrier()
        self.smoothing=smoothing    
        self.epsilion=epsilion
        self.sharp_filtr=get_sharp_filter(k)
        self.k=k

    def __call__(self,feat_i):
        feat_i=self.smoothing(feat_i)
        feat_i-= np.amin(feat_i)
        feat_i/= np.amax(feat_i)
        feat_i+=self.epsilion
        resid_i=np.convolve(feat_i,self.sharp_filtr,mode="valid")
        resized_feat_i=feat_i[self.k:]
        resized_feat_i=resized_feat_i[:-self.k]
        nonlinearity=np.abs(resid_i/resized_feat_i) 
        return [np.mean(nonlinearity),np.median(nonlinearity),np.amax(nonlinearity)]

def basic_stats(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return np.array([np.mean(feat_i),np.std(feat_i),
    	                scipy.stats.skew(feat_i),time_corl(feat_i)])

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)#1.0,step=step)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

def get_sharp_filter(k):
    a=np.ones(k)/(2.0*k)
    return -np.concatenate([a,[-1.0],a])