import numpy as np
import dataset,smooth

def warp_agum(ts_dataset):
    spline_transform=smooth.SplineUpsampling()
    spline_dataset=ts_dataset(spline_transform)
    agum_data=[]
    warp_seq=WrapSeq()
    pos_args=[ [False,False],[False,True],[True,False],[True,True]]
    for name_i in spline_dataset.ts_names():
        data_i=spline_dataset[name_i]
        wrap_i=[  (name_i+'_'+str(i),warp_seq(data_i,*pos_i)) 
                  for i,pos_i in enumerate(pos_args)]
        wrap_i.append( (name_i,data_i))
        agum_data+=wrap_i
    return dataset.TSDataset(dict(agum_data) ,ts_dataset.name+'_warp')

class WrapSeq(object):
    def __init__(self,seq_len=128,sub_seq_len=32):
        self.seq_len=seq_len
        self.start=sub_seq_len
        self.end=seq_len-sub_seq_len
        self.main_spline=smooth.SplineUpsampling(self.seq_len)
        self.long_spline=smooth.SplineUpsampling(2*self.start)
        self.short_spline=smooth.SplineUpsampling(self.start/2)

    def __call__(self,data_i,left=True,short=True):
        return np.array([ self.wrap_feature(feat_i,left,short) 
        	                for feat_i in data_i.T] ).T

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