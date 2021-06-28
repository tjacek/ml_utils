import numpy as np
import json
from dtaidistance import dtw, dtw_ndim
import files,exp,feats

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
        with open('%s.txt' % out_path, 'w') as outfile:
            json.dump(self.pairs, outfile)

def read(in_path):
    pairs= json.load(open("%s" % in_path))
    return DTWpairs(pairs)

def make_dtw_pairs(seqs):
	pairs={ name_i:{name_i:0.0}
				for name_i in seqs.keys()}
	return DTWpairs(pairs)

class Seqs(dict):
	def __init__(self, arg=[]):
		super(Seqs, self).__init__(arg)

	def split(self):
		train,test=files.split(self)
		return Seqs(train),Seqs(test)

def compute_pairs(in_path,out_path=None):
    if(out_path is None):
        out_path=exp.get_out_path(in_path,"pairs")
    seqs=read_seqs(in_path)
    pairs=make_pairwise_distance(seqs)
    pairs.save(out_path)

def read_seqs(in_path):
	seqs=Seqs()
	for path_i in files.top_files(in_path):
		data_i=np.loadtxt(path_i, delimiter=',')
		name_i=path_i.split('/')[-1]
		name_i=files.Name(name_i).clean()#clean(name_i)
		seqs[name_i]=data_i
	return seqs

def make_pairwise_distance(seqs):
	dtw_pairs=make_dtw_pairs(seqs)
	names=list(seqs.keys())
	n_ts=len(names)
	for i in range(1,n_ts):
		print(i)
		for j in range(0,i):
			name_i,name_j=names[i],names[j]
			distance_ij=dtw_ndim.distance(seqs[name_i],seqs[name_j])
			dtw_pairs.set(name_i,name_j,distance_ij)
			dtw_pairs.set(name_j,name_i,distance_ij)
	return dtw_pairs

in_path="../MSR/max_z/seqs"
#compute_pairs(in_path)
out_path="../MSR/max_z/pairs.txt"
dict_i=read(out_path)
feats_i=dict_i.as_feats()
feats_i.save("../MSR/max_z/dtw.txt")