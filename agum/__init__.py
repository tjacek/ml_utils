import numpy as np
import itertools
import dataset,unify,smooth,filtr

def scale_agum(ts_dataset):
    spline_transform=smooth.SplineUpsampling()
    spline_dataset=ts_dataset(spline_transform)    
    train,test=filtr.split( spline_dataset.ts_names())
    agum=[]
    for name_i in train:
        data_i=spline_dataset[name_i]
        for j,scale_j in enumerate([0.5,1.0]):
        	agum.append((name_i+'_'+str(j+4),scale_j*data_i))
    for name_i in test:
        agum.append( (name_i,spline_dataset[name_i]))	
    return dataset.TSDataset(dict(agum) ,ts_dataset.name+'_warp')

#def smooth_agum(ts_dataset):
#    smooth_ts=ts_dataset(smooth.Fourrier())
#    new_dict=smooth_ts.ts_dict.copy()
#    for name_i,data_i in smooth_ts.ts_dict.items():
#        new_dict[name_i+'_agum']=data_i
#    return dataset.TSDataset(new_dict,ts_dataset.name+'_agum')	

#def img_dataset(ts_dataset):
#    spline_ts=ts_dataset(smooth.SplineUpsampling())
#    spline_ts.save()
