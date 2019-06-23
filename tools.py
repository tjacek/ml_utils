import numpy as np

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