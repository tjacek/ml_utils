import numpy as np,random
from scipy.interpolate import CubicSpline
import files,feats,data_dict

class Seqs(data_dict.DataDict):#dict):
    def as_dataset(self):
        X,y=[],[]
        names=list(self.keys())
        for name_i in names:
            X.append(self[name_i])
            y.append(name_i.get_cat())
        return np.array(X),y,names

    def resize(self,new_size=64):
        for name_i in self.keys():
            self[name_i]=inter(self[name_i],new_size)

    def shape(self):
        return [value_i.shape for value_i in self.values()]

    def subsample(self,n):
        def helper(frames):
            indexes=list(range(len(frames)))
            random.shuffle(indexes)
            indexes=indexes[:n]
            indexes.sort()
            return frames[indexes]
        self.transform(helper)

    def save(self,out_path):
        files.make_dir(out_path)
        for name_i,seq_i in self.items():
            print(seq_i.shape)
            out_i="%s/%s" % (out_path,name_i)
            np.save(out_i,seq_i)

def read_seqs(in_path):
    seqs=Seqs()
    for path_i in files.top_files(in_path):
        data_i=read_data(path_i)
        name_i=path_i.split('/')[-1]
        name_i=files.Name(name_i).clean()
        seqs[name_i]=data_i
    return seqs

def selected_read(in_path,cond=None,i=1):
    helper=lambda x:x.split('/')[-1]==cond
    paths=[]
    for path_i in files.top_files(in_path):
        paths+=[ path_j 
            for path_j in files.top_files(path_i)
              if(helper(path_j))]
    return from_paths(paths,i)

def from_paths(paths,i=1):
    seqs=Seqs()
    for path_i in paths:
        data_i=read_data(path_i)
        print(data_i.shape)
        name_i=path_i.split('/')[-i]
        name_i=files.Name(name_i).clean()
        seqs[name_i]=data_i
    return seqs

def read_data(path_i):
    if(is_npy(path_i)):
        return np.load(path_i)
    else:
        return np.loadtxt(path_i, delimiter=',')

def is_npy(path_i):
    return path_i.split(".")[-1]=="npy"

def normalize(ts):
    for ts_i in ts.T:
        ts_i-=np.mean(ts_i)
        ts_i/=np.std(ts_i)
    return ts

def transform_lazy(in_path,fun,out_path=None):
    new_feats=feats.Feats()
    for path_i in files.top_files(in_path):
        data_i=read_data(path_i)
        name_i=files.get_name(path_i)
        new_feats[name_i]=fun(name_i,data_i)
    if(out_path):
        new_feats.save(out_path)
    return new_feats