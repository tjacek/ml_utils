import numpy as np
from scipy.interpolate import CubicSpline
import files

class Seqs(dict):
    def __init__(self, arg=[]):
        super(Seqs, self).__init__(arg)

    def as_dataset(self):
        X,y=[],[]
        names=list(self.keys())
        for name_i in names:
            X.append(self[name_i])
            y.append(name_i.get_cat())
        return np.array(X),y,names

    def split(self):
        train,test=files.split(self)
        return Seqs(train),Seqs(test)

    def resize(self,new_size=64):
        for name_i in self.keys():
            self[name_i]=inter(self[name_i],new_size)

def read_seqs(in_path):
    seqs=Seqs()
    for path_i in files.top_files(in_path):
        data_i=np.loadtxt(path_i, delimiter=',')
        name_i=path_i.split('/')[-1]
        name_i=files.Name(name_i).clean()
        seqs[name_i]=data_i
    return seqs

def inter(ts_i,new_size):
    old_size=ts_i.shape[0]
    step= new_size/old_size
    old_x=np.arange(old_size).astype(float)
    old_x*=step
    cs=CubicSpline(old_x,ts_i)
    new_x=np.arange(new_size)
    return cs(new_x)