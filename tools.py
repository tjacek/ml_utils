import numpy as np
from sklearn.neighbors import KernelDensity

class Fourrier(object):
    def __init__(self):
        self.name="fourrier"

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        return np.abs(rft)

class FourrierNoise(object):
    def __init__(self, n=5):
        self.name="fourrier_noise"
        self.n=n

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        rft[:self.n] = 0
        return np.fft.irfft(rft)

class KernelDist(object):
    def __init__(self):
        self.name="KD"
        self.kernel=KernelDensity(bandwidth=1.0, kernel='gaussian')
        self.n_steps=100

    def __call__(self,feature_i):
        feature_i=feature_i.reshape(-1, 1)
        self.kernel.fit(feature_i)
        max_i,min_i=np.amax(feature_i),np.amin(feature_i)
        step_i=(max_i-min_i)/float(self.n_steps)
        x_i=(np.arange(self.n_steps)*step_i)-min_i        
        x_i=x_i.reshape(-1,1)
        log_dens=self.kernel.score_samples(x_i)
        return np.exp(log_dens)