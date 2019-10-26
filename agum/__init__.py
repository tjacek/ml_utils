import numpy as np
import itertools
import dataset,unify,smooth
import filtr,agum.warp


class AgumProduct(object):
	def __init__(self, arg):
		self.agum_func=agum_func
        self.upsampling=smooth.SplineUpsampling()
		

class AgumSum(object):
    def __init__(self,agum_func):
        self.agum_func=agum_func
        self.upsampling=smooth.SplineUpsampling()

    def __call__(self,ts_dataset):
        train,test=self.prepare(ts_dataset)
        agum=[]
        for name_i,data_i in train:
            agum_data=[]
            for agum_j in self.agum_func:
                agum_data+=agum_j(data_i)
            agum+=[ (name_i+'_'+str(j),agum_j)
                    for j,agum_j in enumerate(agum_data)]
        agum=train+test+agum
        return dataset.TSDataset(dict(agum) ,ts_dataset.name+'_agum')

    def prepare(self,ts_dataset):
        ts_dataset=ts_dataset(self.upsampling)
        train,test=filtr.split(ts_dataset.ts_names())
        train=[ ts_dataset[train_i] for train_i in train]
        test=[ ts_dataset[test_i] for test_i in train]
        return train,test

def scale_agum(data_i):
	return [scale_j*data_i for scale_j in [0.5,2.0]]

def sigma_agum(data_i):
    sigma_i=np.std(data_i)	
    return [data_i+sigma_i,data_i-sigma_i]

def get_warp(type):
    if(type=="scale"):
        return AgumSum([agum.warp.WrapSeq(),scale_agum])
    return AgumSum([agum.warp.WrapSeq()])