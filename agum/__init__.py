import numpy as np
import itertools
import dataset,unify,smooth
import filtr,agum.warp

class AgumSum(object):
    def __init__(self,agum_func):
        self.agum_func=agum_func
        self.upsampling=smooth.SplineUpsampling()

    def __call__(self,ts_dataset):
        ts_dataset=ts_dataset(self.upsampling)
        train,test=filtr.split(ts_dataset.ts_names())
        agum=[]
        for name_i in train:
            data_i=ts_dataset[name_i]
            agum_data=[]
            for agum_j in self.agum_func:
                agum_data+=agum_j(data_i)
            agum+=[ (name_i+'_'+str(j),agum_j)
                    for j,agum_j in enumerate(agum_data)]
        for name_i in ts_dataset.ts_names():
            agum.append( (name_i,ts_dataset[name_i]))
        return dataset.TSDataset(dict(agum) ,ts_dataset.name+'_agum')

def scale_agum(data_i):
	return [scale_j*data_i for scale_j in [0.5,2.0]]

def basic_warp():
    return AgumSum([agum.warp.WrapSeq(),scale_agum])