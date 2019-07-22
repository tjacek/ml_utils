import numpy as np
import itertools
import dataset,unify,smooth,plot,filtr

def show_cut(in_path):
    raw_ts=unify.read(in_path)
    smooth_ts=raw_ts(smooth.Gauss())
    start_ts=smooth_ts(cut_ts,as_array=False)
    plot.plot_by_feat(start_ts)

def img_dataset(ts_dataset):
    spline_ts=ts_dataset(smooth.SplineUpsampling())
    spline_ts.save()

def basic_agum(raw_ts):
    smooth_ts=raw_ts(smooth.Gauss())
    name_by_cat=filtr.train_by_cat(smooth_ts)
    name_pairs=[ filtr.random_pairs(cat_i) for i,cat_i in name_by_cat.items()]
    name_pairs = list(itertools.chain(*name_pairs))
    new_ts=dict([merge_seq(pair_i, smooth_ts) for pair_i in name_pairs])
    return dataset.TSDataset(new_ts,raw_ts.name+"_agum")

def merge_seq(pair_i, ts_data):
    a_feats,b_feats=ts_data.as_features(pair_i[0]),ts_data.as_features(pair_i[1])
    new_ts=[merge_feats(x_i,y_i) for x_i,y_i in zip(a_feats,b_feats)]
    new_name=pair_i[0]+"_agum"
    return new_name,new_ts

def merge_feats(x_i,y_i):
    new_start=cut_feat(y_i)
    new_end=x_i[new_start.shape[0]:,]
    if(new_end.shape[0]>0):
        diff_i=new_start[-1]-new_end[0]
        new_start=new_start- diff_i
    new_feat=np.concatenate([new_start,new_end])
    return new_feat

def cut_feat(feature_i):
    optim_i=local_extr(feature_i)
    if(optim_i.shape[0]<2):
        return feature_i
    return feature_i[:optim_i[1]]

dataset=dtw_agum("mra")
#plot.plot_by_feat(dataset)
#dataset.save()
#img_dataset(dataset)