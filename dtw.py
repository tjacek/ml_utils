import numpy as np
import json
from dtaidistance import dtw, dtw_ndim
import files,exp,feats,seqs,data_dict

class DTWpairs(data_dict.DataDict):
    def __init__(self,args=[]):
        super(DTWpairs, self).__init__(args)
    
    def set(self, key1,key2, value):
        self[key1][key2]=value
    
    def as_feats(self):
        train,test=self.split()
        def helper(name_i):
            return [self[name_i][name_j]
                        for name_j in train]
        dtw_feats=feats.Feats()
        for name_i in self.keys():
            dtw_feats[name_i]=np.array(helper(name_i))
        return dtw_feats   
    
    def neighbors(self,name_i,k=5):
        dict_i=self[name_i].items()
        names,distance=zip(*list(dict_i))
        indexes= np.argsort(distance)
        return [names[i] for i in indexes[:k]]

    def save(self,out_path):
        with open(out_path, 'w') as outfile:
            json.dump(self, outfile)

def read(in_path):
    pairs= json.load(open("%s" % in_path))
    return DTWpairs(pairs)

def make_dtw_pairs(ts):
	pairs={ name_i:{name_i:0.0}
				for name_i in ts.keys()}
	return DTWpairs(pairs)

def compute_pairs(ts,out_path=None,transform=None):
    if(type(ts)==str):
         ts=seqs.read_seqs(in_path)
    if(transform):
        if(transform=='norm'):
            transform=seqs.normalize
    ts.transform(transform)
    pairs=make_pairwise_distance(ts)
    pairs.save(out_path)

def make_pairwise_distance(ts):
	dtw_pairs=make_dtw_pairs(ts)
	names=list(ts.keys())
	n_ts=len(names)
	for i in range(1,n_ts):
		print(i)
		for j in range(0,i):
			name_i,name_j=names[i],names[j]
			distance_ij=dtw_ndim.distance(ts[name_i],ts[name_j])
			dtw_pairs.set(name_i,name_j,distance_ij)
			dtw_pairs.set(name_j,name_i,distance_ij)
	return dtw_pairs

def test_dtw(in_path):
    import learn
    dtw_pairs=read(in_path)
    dtw_feats=dtw_pairs.as_feats()
    result=learn.train_model(dtw_feats)
    result.report()

def exp_dtw(in_path,n=10):
    files.make_dir("dtw")
    for i in range(n):
        seq_dict= seqs.selected_read(in_path,f"{i}.npy",2)
        compute_pairs(seq_dict,f"dtw/test_{i}",transform='norm')
        test_dtw(f"dtw/test_{i}")

if __name__ == "__main__":
    in_path="../CZU-MHAD/test_spline"
    exp_dtw(in_path)