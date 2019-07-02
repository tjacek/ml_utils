import numpy as np 
import scipy.stats

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info

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
    lines=open(in_path,'r').readlines()
    X,info=[],[]
    for line_i in lines:
        data_i,info_i=line_i.split('#')
        X.append(np.fromstring(data_i,sep=','))
        info.append(info_i)
    return FeatureSet(np.array(X),info)

def basic_stats(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0]
    return np.array([np.mean(feat_i),np.std(feat_i),scipy.stats.skew(feat_i)])

print(read('btf.txt').X.shape)