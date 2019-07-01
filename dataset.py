import numpy as np
import files,feats

class TSDataset(object):
    def __init__(self,ts_dict,name="dataset"):
        self.ts_dict=ts_dict
        self.name=name
    
    def __getitem__(self,name_i):
        return self.ts_dict[name_i]

    def __call__(self,transform):
        new_ts_dict={}
        for ts_name_i in self.ts_names():
            feats_i=self.as_features(ts_name_i)
            new_feats=[ transform(feat_ij) for feat_ij in feats_i]
            new_ts_dict[ts_name_i]=np.swapaxes(np.array(new_feats),0,1)
        new_name=self.name+'_'+transform.name
        return TSDataset(new_ts_dict,new_name)

    def ts_names(self):
        return self.ts_dict.keys()

    def as_features(self,name):
        ts_i=self.ts_dict[name]
        return [ts_i[:,j] for j in range(self.n_feats()) ]

    def to_array(self):
        return np.array(self.ts_dict.values())
    
    def n_feats(self):
        ts=self.ts_dict.values()[0]
        return ts.shape[1]

    def to_feats(self,extractor):
        names=self.ts_names()
        def feat_helper(name_i):
            feats_i=[extractor(feat_j) 
                        for feat_j in self.as_features(name_i)]
            return np.array(feats_i).flatten()
        X=np.array([feat_helper(name_i)
                            for name_i in names])
        return feats.FeatureSet(X,names)

def read_dataset(in_path):
    dataset_name=in_path.split("/")[-1]
    paths=files.bottom_files(in_path)
    if(not paths):
        raise Exception("No data at:"+in_path)
    ts_dataset={}
    for path_i in paths:
        ts_i=np.loadtxt(path_i,dtype=float,delimiter=",")
        ts_name_i=path_i.split('/')[-1]
        ts_name_i=ts_name_i.split(".")[0]
        ts_dataset[ts_name_i]=ts_i
    print(dataset_name)
    return TSDataset(ts_dataset,dataset_name)

if __name__ == "__main__":
    read_dataset("seqs/inert")