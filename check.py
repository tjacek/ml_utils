import numpy as np,os
import feats,files,dtw

@files.dir_function
def check_feats(in_path):
    name_i=in_path.split('/')[-1]
    name_i=name_i.split(".")[0]
    feat_dict=feats.read(in_path)[0]
    n_cats=feat_dict.names().n_cats()
    print(f"{name_i},{n_cats},{feat_dict}")

@files.dir_function(args=1)
def get_acc(in_path):
    import dtw
    result_i=dtw.test_dtw(in_path)
    return result_i.get_acc()
#    data_i=feats.read(in_path)[0]
#    print(len(data_i))

in_path="dtw"
acc=get_acc(in_path)
print(acc)