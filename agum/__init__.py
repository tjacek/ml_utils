import numpy as np
#import itertools
import dataset,unify,smooth
import filtr,agum.warp

class Agum(object):
    def __init__(self,agum_func,sum=True):
        self.agum_func=agum_func
        self.upsampling=smooth.SplineUpsampling()
        self.sum=sum

    def __call__(self,ts_dataset):
        train,test=self.prepare(ts_dataset)
        if(self.sum):
            agum=self.sum_agum(train)
            agum=train+test+agum
        else:
            agum=self.product_agum(train)
            agum=test+agum
        return dataset.TSDataset(dict(agum) ,ts_dataset.name+'_agum')

    def sum_agum(self,train):
        agum=[]
        for name_i,data_i in train:
            agum_data=[]
            for agum_j in self.agum_func:
                agum_data+=agum_j(data_i)
            agum+=[ (name_i+'_'+str(j),agum_j)
                    for j,agum_j in enumerate(agum_data)]    	
        return agum

    def product_agum(self,train):
        for func_i in self.agum_func:
            agum_train=[]
            for j,(name_j,data_j) in enumerate(train):
                agum_seq=self.agum_sample(func_i,data_j)
                agum_train+=[ (name_j+'_'+str(k),data_k)
                                for k,data_k in enumerate(agum_seq)]  
            train=agum_train           
        return train

    def agum_sample(self,func_i,data_j):
        agum_ts=[ func_i(data_t) for data_t in data_j.T]
        agum_ts=list(zip(*agum_ts))
        return [ np.array(ts).T for ts in agum_ts]

    def prepare(self,ts_dataset):
        ts_dataset=ts_dataset(self.upsampling)
        train,test=filtr.split(ts_dataset.ts_names())
        train=[ (train_i,ts_dataset[train_i]) for train_i in train]
        test=[ (test_i,ts_dataset[test_i]) for test_i in test]
        return train,test

def get_warp(type,sum=True):
    if(type=="scale"):
        return Agum([agum.warp.WrapSeq(),scale_agum],sum=sum)
    return Agum([agum.warp.WrapSeq()],sum=sum)

def scale_agum(data_i):
	return [scale_j*data_i for scale_j in [0.5,2.0]]

def sigma_agum(data_i):
    sigma_i=np.std(data_i)	
    return [data_i+sigma_i,data_i-sigma_i]

def gauss_agum(n=4):
    if(n<2):
        return Agum([agum.warp.WrapSeq()])
    kerns=[]
    for sigma_i in range(1,n):
        x_i=np.arange(-3*sigma_i, 3*sigma_i, 1.0)
        kerns.append(np.exp( -(x_i/sigma_i)**2/2) )
    def gauss_helper(data_i):
        smooth_seq=[np.convolve(data_i,kern_i,mode="same") for kern_i in kerns]
        smooth_seq.append(data_i)
        return smooth_seq
    return Agum([agum.warp.WrapSeq(),gauss_helper],sum=False)