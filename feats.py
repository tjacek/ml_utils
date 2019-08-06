import files
import numpy as np 
import os,re
from sets import Set
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info
    
    def __add__(self,feat_i):
        new_X=np.concatenate([self.X,feat_i.X],axis=1)
        return FeatureSet(new_X,self.info)

    def dim(self):
        return self.X.shape[1]

    def get_labels(self):
        return [ int(info_i.split('_')[0]) for info_i in self.info]

    def to_dict(self):
        return { self.info[i]:x_i 
                    for i,x_i in enumerate(self.X)}

    def norm(self):
        self.X=preprocessing.scale(self.X)
    
    def reduce(self,n=100):
        if(self.dim()>n and n!=0):
            svc = SVC(kernel='linear',C=1)
            rfe = RFE(estimator=svc,n_features_to_select=n,step=10)
            rfe.fit(self.X,self.get_labels())
            #print(rfe.ranking_)
            self.X= rfe.transform(self.X)
        return self

    def save(self,out_path,decimals=4):
        lines=[ np.array2string(x_i,separator=",",precision=decimals) for x_i in self.X]
        lines=[ line_i.replace('\n',"")+'#'+info_i 
                    for line_i,info_i in zip(lines,self.info)]
        feat_txt='\n'.join(lines)
        feat_txt=feat_txt.replace('[','').replace(']','')
        file_str = open(out_path,'w')
        file_str.write(feat_txt)
        file_str.close()

def read(in_path):
    data_dict=unify_dict(in_path) if(os.path.isdir(in_path)) else read_single(in_path)
    return from_dict(data_dict)
#    if(os.path.isdir(in_path)):
#        new_X=np.concatenate([data_i.X for data_i in datasets],axis=1)
#        return FeatureSet(new_X,datasets[0].info)
#    else:
#        return from_dict(read_single(in_path))

def unify_dict(in_path):
    datasets=[ read_single(path_i) for path_i in files.top_files(in_path)]
    name_sets=[ Set(data_i.keys()) for data_i in datasets ]
    common_names=name_sets[0]
    for set_i in name_sets[1:]:
        common_names=common_names.intersection(set_i)
    def unify_helper(name_i):
        return np.concatenate([data_i[name_i] for data_i in datasets])
    return { name_i:unify_helper(name_i) for name_i in common_names}

def read_single(in_path):
    lines=open(in_path,'r').readlines()
    feat_dict={}
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.clean_str(info_i)#re.sub(r'[a-z]','',info_i.strip())
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict

def from_dict(feat_dict):
    info=files.natural_sort(feat_dict.keys())
    X=np.array([feat_dict[info_i] 
                    for info_i in info])
    return FeatureSet(X,info)