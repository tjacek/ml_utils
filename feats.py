import numpy as np
import os.path
from sklearn import preprocessing
import files

class Feats(dict):
    def __init__(self, arg=[]):
        super(Feats, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(Feats, self).__setitem__(key, value)

    def dim(self):
        return list(self.values())[0].shape

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return Feats(train),Feats(test)

    def as_dataset(self):
        names=self.names()
        return self.get_X(names),self.get_labels(names),names

    def names(self):
        return files.NameList(self.keys())
#        return sorted(self.keys(),key=files.natural_keys) 
    
    def get_X(self,names=None):
        if(names is None):
            names=self.names()
        return np.array([self[name_i] for name_i in names])

    def get_labels(self,names=None):
        if(names is None):
            names=self.names()
        return [ name_i.get_cat() for name_i in names]

    def __add__(self,feat_i):
        names=common_names(self.keys(),feat_i.keys())
        names.sort()
        new_feats=Feats()
        for name_i in names:
            x_i=np.concatenate([self[name_i],feat_i[name_i]],axis=0)
            new_feats[name_i]=x_i
        return new_feats

    def rename(self,name_dict):
        new_feats=Feats()
        for name_i,name_j in name_dict.items():
            new_feats[files.Name(name_j)]=self[name_i]
        return new_feats

    def norm(self):
        names=self.names()
        X=np.array([self[name_i] for name_i in names])
        new_X=preprocessing.scale(X)
        for i,name_i in enumerate(names):
            self[name_i]=new_X[i]

    def save(self,out_path,decimals=10):
        lines=[]
        for name_i,x_i in self.items():
            str_i=np.array2string(x_i,separator=",",precision=decimals)
            line_i="%s#%s" % (str_i.replace('\n',""),name_i)
            lines.append(line_i)
        feat_txt='\n'.join(lines)
        feat_txt=feat_txt.replace('[','').replace(']','')
        with open(out_path,'w') as file_str:
            file_str.write(feat_txt)

    def subset(self,names,new_names=False):
        sub_dict=Feats()
        for i,name_i in enumerate( names):
            value_i=self[name_i]
            if(new_names):
                name_i=f'{name_i}_{i}'
            sub_dict[name_i]=value_i
        return sub_dict 

    def transform(self,fun,copy=False):
        new_dict= Feats() if(copy) else self
        for name_i,data_i in self.items():
            new_dict[name_i]=fun(data_i)
        return new_dict

    def __str__(self):
        return "%d,%d"  % (len(self),self.dim()[0])

def read(in_path):
    if(type(in_path)==list):
        return [read_unified(in_path)]
    if(not os.path.isdir(in_path)):
        return [read_single(in_path)]
    return [read_single(path_i) 
                for path_i in files.top_files(in_path)]

def read_single(in_path):
    lines=open(in_path,'r').readlines()
    feat_dict={}
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.Name(info_i).clean()
            x_i=np.fromstring(data_i,sep=',')
            x_i=np.nan_to_num(x_i,nan=0.0,posinf=0.0, neginf=0.0)
            feat_dict[info_i]=x_i
    return Feats(feat_dict)

def read_unified(paths):
    datasets=[read_single(path_i) 
                for path_i in paths]
    full_data=datasets[0]
    for data_i in datasets[1:]:
        full_data+=data_i
    return full_data

def common_names(names1,names2):
    return list(set(names1).intersection(set(names2)))

def agum_concat(data_a,data_b,selector=None):
    if(selector is None):
        selector=files.person_selector
    if(len(data_a)<len(data_b)):
        base_data,agum_data=data_a,data_b
    else:
        base_data,agum_data=data_b,data_a
    unified_data=Feats()
    for name_i,data_i in agum_data.items():
        if(len(name_i)>4):
            id_i=name_i.subname(-1)
        else:
            id_i=name_i
        feat_i=np.concatenate( [data_i,base_data[id_i]] )
        if(selector(name_i)):
            unified_data[name_i]=feat_i
        else:
            unified_data[id_i]=feat_i
    return unified_data