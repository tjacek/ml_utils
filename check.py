import numpy as np,os
import feats,files

@files.dir_function
def check_feats(in_path):
    name_i=in_path.split('/')[-1]
    name_i=name_i.split(".")[0]
    feat_dict=feats.read(in_path)[0]
    n_cats=feat_dict.names().n_cats()
    print(f"{name_i},{n_cats},{feat_dict}")

in_path="../data/I/common"
check_feats(in_path)