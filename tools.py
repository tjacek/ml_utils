import numpy as np

class Fourrier(object):
    def __init__(self):
        self.name="fourrier"

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        return np.abs(rft)