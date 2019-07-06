import files
import numpy as np 
import os,re
from sklearn import preprocessing

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info
    
    def get_labels(self):
        return [ int(info_i.split('_')[0]) for info_i in self.info]

    def to_dict(self):
        return { self.info[i]:x_i 
                    for i,x_i in enumerate(self.X)}

    def norm(self):
        self.X=preprocessing.scale(self.X)
    
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
    if(os.path.isdir(in_path)):
        datasets=[ read_single(path_i) for path_i in files.top_files(in_path)]
        new_X=np.concatenate([data_i.X for data_i in datasets],axis=1)
        return FeatureSet(new_X,datasets[0].info)
    else:
        return read_single(in_path)

def read_single(in_path):
    lines=open(in_path,'r').readlines()
    X,info=[],[]
    for line_i in lines:
        data_i,info_i=line_i.split('#')
        info_i=re.sub(r'[a-z]','',info_i.strip())
        X.append(np.fromstring(data_i,sep=','))
        info.append(info_i)
    return FeatureSet(np.array(X),info)

def from_dict(feat_dict):
    info=feat_dict.keys()
    X=np.array([feat_dict[info_i] 
                    for info_i in info])
    return FeatureSet(X,info)