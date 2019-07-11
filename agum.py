import numpy as np
import itertools
import dataset,unify,smooth,plot,filtr

def dtw_agum(in_path):
    raw_ts=unify.read(in_path)
    smooth_ts=raw_ts(smooth.Gauss())
    name_by_cat=filtr.train_by_cat(smooth_ts)
    name_pairs=[ filtr.random_pairs(cat_i) for i,cat_i in name_by_cat.items()]
    name_pairs = list(itertools.chain(*name_pairs))
    new_ts=dict([merge_seq(pair_i, smooth_ts) for pair_i in name_pairs])
    return dataset.TSDataset(new_ts,raw_ts.name+"_agum")

def show_cut(in_path):
    raw_ts=unify.read(in_path)
    smooth_ts=raw_ts(smooth.Gauss())
    start_ts=smooth_ts(cut_ts,as_array=False)
    plot.plot_by_feat(start_ts)

def merge_seq(pair_i, ts_data):
    a_feats,b_feats=ts_data.as_features(pair_i[0]),ts_data.as_features(pair_i[1])
    new_ts=[merge_feats(x_i,y_i) for x_i,y_i in zip(a_feats,b_feats)]
    new_name=pair_i[0]+"_agum"
    return new_name,new_ts

def merge_feats(x_i,y_i):
    new_start=cut_ts(y_i)
    new_end=x_i[y_i.shape[0]:]
    new_feat=np.concatenate([new_start,new_end])
    return new_feat

def cut_ts(feature_i):
    optim_i=np.where(local_extr(feature_i)!=0)[0]
    if(optim_i.shape[0]<2):
        return feature_i
    return feature_i[:optim_i[1]]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))

plot.plot_by_feat( dtw_agum("mra"))