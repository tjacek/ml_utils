import numpy as np
import scipy.signal
from scipy.interpolate import CubicSpline

class SplineUpsampling(object):
    def __init__(self,new_size=128):
        self.name="spline"
        self.new_size=new_size

    def __call__(self,feat_i):
        old_size=feat_i.shape[0]
        old_x=np.arange(old_size)
        old_x=old_x.astype(float)  
        step=float(self.new_size)/float(old_size)
        old_x*=step     
        cs=CubicSpline(old_x,feat_i)
        new_x=np.arange(self.new_size)
        return cs(new_x)

class Gauss(object):
    def __init__(self, window=10,sigma=5.0):
        self.name="gauss"
        self.filtr=scipy.signal.gaussian(window,sigma)

    def __call__(self,feature_i):
        return scipy.signal.convolve(feature_i,self.filtr,mode='same')	

class FourrierNoise(object):
    def __init__(self, n=5):
        self.name="fourrier_noise"
        self.n=n

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        rft[:self.n] = 0
        return np.fft.irfft(rft)

class ResPairs(object):
    def __init__(self):
        self.name="ResSmooth"
        self.residuals=FourrierNoise()

    def __call__(self,feature_i):
        residuals_i=self.residuals(feature_i)
        feature_i=feature_i[:len(residuals_i)]
        smooth_i=feature_i-residuals_i
        pairs_i=np.array([residuals_i,smooth_i]).T
        return pairs_i