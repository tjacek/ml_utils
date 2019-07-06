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

def non_linear(feat_i):
    feat_i=np.fft.rfft(feat_i)
    feat_i[:5] = 0
    feat_i=np.fft.irfft(feat_i)
    feat_i-= np.amin(feat_i)
    feat_i/= np.amax(feat_i)
    feat_i+=0.01
    filtr=np.array([-0.25,-0.25,1.0,-0.25,-0.25])
    resid_i=np.convolve(feat_i,filtr,mode="valid")
    resized_feat_i=feat_i[2:]
    resized_feat_i=resized_feat_i[:-2]
    nonlinearity=np.abs(resid_i/resized_feat_i) 
    return [np.mean(nonlinearity),np.median(nonlinearity),np.amax(nonlinearity)]