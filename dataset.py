import numpy as np
import read

class TSDataset(object):
    def __init__(self,ts_dict,name="dataset"):
        self.ts_dict=ts_dict
        self.name=name

    def ts_names(self):
        return self.ts_dict.keys()

    def as_features(self,name):
        return [ x_i for x_i in self.ts_dict[name].T]
    
    def __call__(self,transform):
        new_ts_dict={}
        for ts_name_i in self.ts_names():
            feats_i=self.as_features(ts_name_i)
            new_feats=[ transform(feat_ij) for feat_ij in feats_i]
            new_ts_dict[ts_name_i]=np.array(new_feats).T
        new_name=self.name+'_'+transform.name
        return TSDataset(new_ts_dict,new_name)

def read_dataset(in_path):
    dataset_name=in_path.split("/")[-1]
    paths=read.bottom_files(in_path)
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