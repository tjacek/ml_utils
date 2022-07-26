import numpy as np
import scipy.signal
from scipy.interpolate import CubicSpline
import files,seqs

class SplineUpsampling(object):
    def __init__(self,new_size=64):
        self.name="spline"
        self.new_size=new_size

    def __call__(self,feat_i):
        old_size=feat_i.shape[0]
        old_x=np.arange(old_size)
        old_x=old_x.astype(float)  
        if(self.new_size):
            step=float(self.new_size)/float(old_size)
            old_x*=step     
            cs=CubicSpline(old_x,feat_i)
            new_size=np.arange(self.new_size)  
            return cs(new_size)
        else:
            cs=CubicSpline(old_x,feat_i)
            return cs(old_x)

@files.dir_function(args=2,recreate=False)
@files.dir_function(args=2,recreate=True)
def upsample(in_path,out_path):
    print(in_path)
    seq_i=seqs.read_data(in_path)
    seq_i=SplineUpsampling()(seq_i)
    np.save(out_path,seq_i)

in_path="../CZU-MHAD/inert/"#qyh_a12_t6.mat"
upsample(in_path,"../CZU-MHAD/spline")