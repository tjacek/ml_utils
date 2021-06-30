import numpy as np,os
import feats

def check_feats(in_path):
    feat_dict=feats.read(in_path)[0]
    print("size:%d,dim:%d" % (len(feat_dict),feat_dict.dim()[0]))

check_feats("../MHAD/max_z/dtw/feats")