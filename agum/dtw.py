import numpy as np,itertools
import dataset,filtr,smooth,unify,dtw

def dtw_agum(in_path):
    raw_ts=unify.read(in_path)
    raw_ts.normalize()
    smooth_ts=raw_ts(smooth.Gauss())
    smooth_ts.save()
    start_ts=smooth_ts(cut_ts,whole_seq=True)
    name_by_cat=filtr.train_by_cat(start_ts)
    name_pairs=[ dtw_pairs(cat_i,start_ts) for i,cat_i in name_by_cat.items()]
    name_pairs = list(itertools.chain(*name_pairs))
    print(name_pairs)
    new_ts=dict([ (pair_i[0]+'_agum',merge_ts(pair_i,smooth_ts)) for pair_i in name_pairs])
    return dataset.TSDataset(new_ts,raw_ts.name+"_agum")

def dtw_pairs(cat_i,data_ts):
    cat_seq=[data_ts.ts_dict[cat_ij] for cat_ij in cat_i]
    metric_arr=dtw.metric_matrix(cat_seq)
    indexes=np.argmin(metric_arr,axis=0)
    return [(cat_i[j],cat_i[k]) for j,k in enumerate(indexes)]

def merge_ts(pair_i,ts_dataset):
    x_i=np.array(ts_dataset.ts_dict[pair_i[0]])
    y_i=np.array(ts_dataset.ts_dict[pair_i[1]])
    new_start=cut_ts(y_i)
    new_end=x_i[new_start.shape[0]:,]
    diff_i= new_start[-1]-new_end[0]
    new_start=np.array([start_j-diff_i[j] 
                for j,start_j in enumerate(new_start.T)]).T
    return np.concatenate([new_start,new_end])

def cut_ts(ts_i):
    optim_i=[ local_extr(feat_i) for feat_i in ts_i]
    cut_size=max([ 0 if(optim_ij.shape[0]<2) else optim_ij[1] 
                        for optim_ij in optim_i])
    return ts_i[:cut_size]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    sign_diff_i=np.diff( np.sign(diff_i))
    return np.where(sign_diff_i!=0)[0]