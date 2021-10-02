import numpy as np,os
import feats

def check_feats(in_path):
    if(type(in_path)==tuple):
        common,binary=in_path
        check_ens(common,binary)
        return
    feat_dict=feats.read(in_path)[0]
    print("size:%d,dim:%d" % (len(feat_dict),feat_dict.dim()[0]))

def check_ens(common,binary):
    common=feats.read(common)[0].dim()[0]
    binary=feats.read(binary)[0].dim()[0]
    print("common:%d,binary:%d" % (common,binary))
    print("sum:%d" % (common+binary))

dir_path="../3DHOI/"
binary_path="%s/ens/I/feats" % dir_path
base_path="%s/1D_CNN/feats" % dir_path
dtw_path="../deep_dtw/dtw"
ae_path="../best2/3_layers/feats"

check_feats(([base_path,dtw_path],binary_path))