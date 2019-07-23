import numpy as np
import itertools
import dataset,unify,smooth

def smooth_agum(ts_dataset):
    smooth_ts=ts_dataset(smooth.Fourrier())
    new_dict=smooth_ts.ts_dict.copy()
    for name_i,data_i in smooth_ts.ts_dict.items():
        new_dict[name_i+'_agum']=data_i
    return dataset.TSDataset(new_dict,ts_dataset.name+'_agum')	

def img_dataset(ts_dataset):
    spline_ts=ts_dataset(smooth.SplineUpsampling())
    spline_ts.save()