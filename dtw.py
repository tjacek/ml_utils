import numpy as np
import json
from dtaidistance import dtw, dtw_ndim
import files,exp,feats,seqs

class DTWpairs(object):
    def __init__(self,pairs):
        self.pairs=pairs

    def keys(self):
        return self.pairs.keys()

    def as_feats(self):
        train,test=self.split()
        def helper(name_i):
            return [self.pairs[name_i][name_j]
                        for name_j in train]
        dtw_feats=feats.Feats()
        for name_i in self.keys():
            dtw_feats[name_i]=np.array(helper(name_i))
        return dtw_feats

    def set(self,key1,key2,data_i):
        self.pairs[key1][key2]=data_i
	
    def split(self,selector=None):
        return files.split(self.pairs,selector,pairs=False)

    def save(self,out_path):
        with open(out_path, 'w') as outfile:
            json.dump(self.pairs, outfile)

def read(in_path):
    pairs= json.load(open("%s" % in_path))
    return DTWpairs(pairs)

def make_dtw_pairs(ts):
	pairs={ name_i:{name_i:0.0}
				for name_i in ts.keys()}
	return DTWpairs(pairs)

def compute_pairs(in_path,out_path=None):
    if(out_path is None):
        out_path=exp.get_out_path(in_path,"pairs")
    ts=seqs.read_seqs(in_path)
    pairs=make_pairwise_distance(ts)
    pairs.save(out_path)
    feat_path=exp.get_out_path(in_path,"dtw")
    dtw_feats=pairs.as_feats()
    dtw_feats.save(feat_path)

def make_pairwise_distance(ts):
	dtw_pairs=make_dtw_pairs(ts)
	names=list(ts.keys())
	n_ts=len(names)
	for i in range(1,n_ts):
		print(i)
		for j in range(0,i):
			name_i,name_j=names[i],names[j]
			distance_ij=dtw_ndim.distance(seqs[name_i],seqs[name_j])
			dtw_pairs.set(name_i,name_j,distance_ij)
			dtw_pairs.set(name_j,name_i,distance_ij)
	return dtw_pairs