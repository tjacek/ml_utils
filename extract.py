import numpy as np
import scipy.stats
#import smooth

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

class NoiseCorl(object):
    def __init__(self, smoothing=None):
        if(not smoothing):
            smoothing=smooth.Fourrier()
        self.res_points= res.ResPoints(smoothing)

    def __call__(self,feat_i):
        res_i=self.res_points(feat_i).T
        return scipy.stats.pearsonr(res_i[0],res_i[1])[0]

class BasicStats(object):
    def __init__(self,stats=None):
        self.stats=stats

    def __call__(self,feat_i):
        if(np.all(feat_i==0)):
            return np.zeros((len(self.stats),))
        return np.array([stat_j(feat_i) 
                            for stat_j in self.stats])
def get_basic_stats():
    return BasicStats([np.mean,np.std,scipy.stats.skew,time_corl])

def get_kurt_stats():
    return BasicStats([np.mean,np.std,scipy.stats.skew,
                        time_corl,scipy.stats.kurtosis])

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)#1.0,step=step)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

def get_sharp_filter(k):
    a=np.ones(k)/(2.0*k)
    return -np.concatenate([a,[-1.0],a])
