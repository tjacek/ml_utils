import numpy as np
import unify,smooth,plot,filtr

def dtw_agum(in_path):
    raw_ts=unify.read(in_path)
    smooth_ts=raw_ts(smooth.Gauss())
    start_ts=smooth_ts(cut_ts,as_array=False)
    name_by_cat=filtr.train_by_cat(start_ts)
    print(name_by_cat)

def show_cut(in_path):
    raw_ts=unify.read(in_path)
    smooth_ts=raw_ts(smooth.Gauss())
    start_ts=smooth_ts(cut_ts,as_array=False)
    plot.plot_by_feat(start_ts)

def cut_ts(feature_i):
    optim_i=np.where(local_extr(feature_i)!=0)[0]
    if(optim_i.shape[0]<2):
        return feature_i
    return feature_i[:optim_i[1]]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))

dtw_agum("mra")