import numpy as np
import dataset,smooth,filtr

class WrapSeq(object):
    def __init__(self,seq_len=128,sub_seq_len=32):
        self.seq_len=seq_len
        self.start=sub_seq_len
        self.end=seq_len-sub_seq_len
        self.main_spline=smooth.SplineUpsampling(self.seq_len)
        self.long_spline=smooth.SplineUpsampling(2*self.start)
        self.short_spline=smooth.SplineUpsampling(self.start/2)

    def __call__(self,data_i):
        agum=[]
        for left in [True,False]:
            for short in [True,False]:
                agum.append(self.wrap_feature(data_i,left,short))                
        return agum

    def wrap_feature(self,data_i,left,short):
        if(left):
            sub_i,rest_i=data_i[:self.start],data_i[self.start:]
        else:
            sub_i,rest_i=data_i[self.end:],data_i[:self.end]
        if(short):
            sub_i=	self.short_spline(sub_i)
        else:
            sub_i=	self.long_spline(sub_i)	
        if(left):
            new_data_i=np.concatenate([sub_i,rest_i])
        else:
            new_data_i=np.concatenate([rest_i,sub_i])
        return self.main_spline(new_data_i)