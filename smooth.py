import numpy as np
import scipy.signal

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