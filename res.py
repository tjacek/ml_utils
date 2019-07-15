import numpy as np
import smooth

class Residuals(object):
    def __init__(self, smoothing=None):
        self.name="res"
        if(not smoothing):
            smoothing=smooth.Gauss()
        self.smoothing=smoothing

    def __call__(self,feature_i):
        smooth_i=self.smoothing(feature_i)
        feature_i=feature_i[:len(smooth_i)]
        return feature_i-smooth_i        

class ResPoints(object):
    def __init__(self,smoothing=None):
        self.name="res_points"
        if(not smoothing):
            smoothing=smooth.Gauss()
        self.smoothing=smoothing

    def __call__(self,feature_i):
        smooth_i=self.smoothing(feature_i)
        feature_i=feature_i[:len(smooth_i)]
        residuals_i=feature_i-smooth_i
        pairs_i=np.array([residuals_i,smooth_i]).T
        return pairs_i