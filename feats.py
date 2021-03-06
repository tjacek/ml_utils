import files,filtr
import numpy as np 
import os,re
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info

    def __len__(self):
        return len(self.info)

    def __add__(self,feat_i):
        if(self.X.shape[0]!=feat_i.X.shape[0]):
            new_info=list(set(self.info).intersection(set(feat_i.info)))
            new_info.sort()
            a_dict,b_dict=self.to_dict(),feat_i.to_dict()
            a_dict={ name_i:a_dict[name_i] for name_i in new_info}
            b_dict={ name_i:b_dict[name_i] for name_i in new_info}
            new_X=np.concatenate([from_dict(a_dict).X,from_dict(b_dict).X],axis=1)
            return  FeatureSet(new_X,new_info)
        new_X=np.concatenate([self.X,feat_i.X],axis=1)
        return FeatureSet(new_X,self.info)

    def dim(self):
        return self.X.shape[1]

    def n_cats(self):
        return len(set(self.get_labels()))

    def get_labels(self):
        return [ int(info_i.split('_')[0])-1 for info_i in self.info]

    def labels_array(self):
        n_cats=self.n_cats()
        cats=self.get_labels()
        arr=[]
        for cat_i in cats:
            one_hot_i=np.zeros((n_cats,))
            one_hot_i[cat_i-1]=1
            arr.append(one_hot_i)
        return np.array(arr)

    def to_dict(self):
        return { self.info[i]:x_i 
                    for i,x_i in enumerate(self.X)}

    def norm(self):
        self.X=preprocessing.scale(self.X)
    
    def remove_nan(self):
        self.X[np.isnan(self.X)]=0.0

    def reduce(self,n=100):
        if(self.dim()>n and n!=0):
            print("Old dim %d" % self.dim())
            svc = SVC(kernel='linear',C=1)
            rfe = RFE(estimator=svc,n_features_to_select=n,step=10)
            rfe.fit(self.X,self.get_labels())
            self.X= rfe.transform(self.X)
            print("New dim %d" % self.dim())
        return self
    
    def split(self,selector=None):
        feat_dict=self.to_dict()
        train,test=filtr.split(feat_dict,selector)
        return from_dict(train),from_dict(test)

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
    dataset_paths=get_paths(in_path)
    return from_dict(unify_dict(dataset_paths))

def get_paths(in_path):
    if(type(in_path)==list):
        dataset_paths=[]
        for path_i in in_path:
            dataset_paths+=get_paths(path_i)
        return dataset_paths
    if(os.path.isdir(in_path)):
        return files.top_files(in_path)
    return [in_path]

def unify_dict(paths):
    if(len(paths)==1):
        return read_single(paths[0])
    datasets=[ read_single(path_i) for path_i in paths]
    name_sets=[ set(data_i.keys()) for data_i in datasets ]
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
            info_i= files.clean_str(info_i)
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict

def from_dict(feat_dict):
    info=files.natural_sort(feat_dict.keys())
    X=np.array([feat_dict[info_i] 
                    for info_i in info])
    return FeatureSet(X,info)

def read_list(in_path):
    if(not os.path.isdir(in_path)):
        return [from_dict(read_single(in_path))]
    return [from_dict(read_single(path_i)) 
                for path_i in files.top_files(in_path)]

def unify(datasets):
    if(len(datasets)<2):
        return datasets[0]
    unified_dict=datasets[0]
    for data_i in datasets[1:]:
        unified_dict+=data_i
    return unified_dict