import numpy as np
import cv2
import re
import files,feats

class TSDataset(object):
    def __init__(self,ts_dict,name="dataset"):
        self.ts_dict=ts_dict
        self.name=name
    
    def __getitem__(self,name_i):
        return self.ts_dict[name_i]

    def __call__(self,transform,as_array=True,whole_seq=False):
        new_ts_dict={}
        for ts_name_i in self.ts_names():
            if(whole_seq):

                new_ts_dict[ts_name_i]=transform(self.ts_dict[ts_name_i])
            else:
                new_ts_dict[ts_name_i]=self.feat_transform(ts_name_i,transform,as_array)
        return TSDataset(new_ts_dict,self.get_name(transform))
 
    def feat_transform(self,ts_name_i,transform,as_array):
        feats_i=self.as_features(ts_name_i)
        new_feats=[ transform(feat_ij) for feat_ij in feats_i]
        if(as_array):
            return np.swapaxes(np.array(new_feats),0,1)
        else:
            return new_feats

    def ts_names(self):
        return self.ts_dict.keys()

    def as_features(self,name):
        ts_i=self.ts_dict[name]
        if(type(ts_i)==list):
            return ts_i
        return [ts_i[:,j] for j in range(self.n_feats()) ]

    def to_array(self):
        return np.concatenate([ ts_i 
                for ts_i in self.ts_dict.values()],axis=0)
    
    def n_feats(self):
        ts=self.ts_dict.values()[0]
        if(type(ts)==list):
            return len(ts)
        return ts.shape[1]

    def to_feats(self,extractor,prec=4):
        names=self.ts_names()
        def feat_helper(name_i):
            feats_i=[extractor(feat_j) 
                        for feat_j in self.as_features(name_i)]
            return np.array(feats_i).flatten()
        X=np.array([feat_helper(name_i)
                            for name_i in names])
        if(type(prec)==int):
            X=np.round(X,prec)
        return feats.FeatureSet(X,names)

    def feat_corl(self):
        return np.corrcoef(self.to_array().T)

    def get_name(self,transform):
        if hasattr(transform,'name'):
            trans_name=transform.name
        else:
            trans_name=transform.__name__
        return self.name+'_'+trans_name

    def save(self):
        files.make_dir(self.name)
        for name_i,data_i in self.ts_dict.items():
            np.savetxt(self.name+'/'+name_i,data_i,fmt='%.4e', delimiter=',')

    def select(self,names):
        new_dict={ name_i:self.ts_dict[name_i] for name_i in names}
        return TSDataset(new_dict,self.name)
    
    def normalize(self):
        feat_means=np.mean(self.to_array(),axis=0)
        for name_i in self.ts_names():
            self.ts_dict[name_i]=[feat_i /feat_means[i]
                    for i,feat_i in enumerate(self.as_features(name_i))]

def read_dataset(in_path):
    dataset_name=in_path.split("/")[-1]
    paths=files.bottom_files(in_path)
    if(not paths):
        raise Exception("No data at:"+in_path)
    ts_dataset={}
    for path_i in paths:
        ts_i=np.loadtxt(path_i,dtype=float,delimiter=",")
        ts_name_i=path_i.split('/')[-1]
        ts_name_i=files.clean_str( ts_name_i)
        print(ts_name_i)
        ts_dataset[ts_name_i]=ts_i
    print(dataset_name)
    return TSDataset(ts_dataset,dataset_name)

def as_imgs(ts_dataset,out_path=None):
    dir_name=ts_dataset.name+'_img'
    if(out_path):
        dir_name=out_path+'/'+dir_name
    files.make_dir(dir_name)
    for name_i in ts_dataset.ts_names():
        img_i=ts_dataset[name_i]
        img_i*=10.0
        out_i=dir_name+'/'+name_i+".png"
        cv2.imwrite(out_i,img_i)

if __name__ == "__main__":
    read_dataset("seqs/inert")