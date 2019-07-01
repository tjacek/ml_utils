import numpy as np 
import scipy.stats

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info

    def save(self,out_path,decimals=4):
        lines=[ np.array2string(x_i,separator=",",precision=decimals) for x_i in self.X]
        lines=[ line_i+'#'+info_i 
                    for line_i,info_i in zip(lines,self.info)]
        feat_txt='\n'.join(lines)
        feat_txt=feat_txt.replace('[','').replace(']','')
        file_str = open(out_path,'w')
        file_str.write(feat_txt)
        file_str.close()

def basic_stats(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0]
    return np.array([np.mean(feat_i),np.std(feat_i),scipy.stats.skew(feat_i)])