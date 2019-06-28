import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats

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
        self.n_steps=100

    def __call__(self,feature_i):
        feature_i=feature_i.reshape(-1, 1)
        kde=self.get_kde(feature_i)
        max_i,min_i=np.amax(feature_i),np.amin(feature_i)
        print(max_i,min_i)
        step_i=(max_i-min_i)/float(self.n_steps)
        x_i=(np.arange(self.n_steps)*step_i)+min_i        
        x_i=x_i.reshape(-1,1)
        log_dens=kde.score_samples(x_i)
        return np.exp(log_dens)

    def get_kde(self,feature_i):
        sd_i=np.std(feature_i)
        n=float(feature_i.shape[0])
        bandwidth_i=(1.06*sd_i)/(n**0.2)
        kde_i=KernelDensity(bandwidth=bandwidth_i, kernel='gaussian')
        kde_i.fit(feature_i)
        return kde_i

class NormTest(object):
    def __init__(self, alpha=0.05):
        self.name="test"
        self.alpha=alpha
  
    def __call__(self,feature_i):
        k2,p=stats.normaltest(feature_i)
        return int(p>self.alpha)

class Autocorl(object):
    def __init__(self):
        self.name="autocorl"

    def __call__(self,feature_i):
        magnitude_i=np.abs(np.fft.fft(feature_i)**2)
        r2=np.fft.ifft(magnitude_i).real
        c=(r2/feature_i.shape-np.mean(feature_i)**2)/np.std(feature_i)**2
        return c[:len(feature_i)//2]
