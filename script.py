import numpy as np
import learn,files,exp

class ConstCommon(object):
    def __init__(self,base_path,name="ae_"):
        self.base_path=base_path
        self.name=name

    def __call__(self,input_dict):
        common,binary=input_dict
        for common_i in common:
            desc_i="%s_%s" % (self.name,common_i.split("/")[-1]) 
            common_i=[self.base_path,common_i]
            yield desc_i,(common_i,binary)

def prepare_paths(dir_path,dtw="dtw",nn="1D_CNN",binary="ens_splitI"):
    common="%s/%s" % (dir_path,dtw)
    common=files.get_paths(common,name="dtw")
    if(nn):
        common.append("%s/%s/feats" % (dir_path,nn))
    binary="%s/%s/feats" % (dir_path,binary)
    return {"common":common,"binary":binary}

if __name__ == "__main__":
    dir_path="../3DHOI/"
    binary_path="%s/ens/I/feats" % dir_path
    base_path="%s/1D_CNN/feats" % dir_path
    dtw_path="../deep_dtw/dtw"
    ae_path="../best2/3_layers/feats"
    common=[dtw_path,ae_path]
    helper=ConstCommon(base_path)
    ens_exp=exp.EnsembleExp(gen=helper)
    ens_exp([common,binary_path],out_path=None)