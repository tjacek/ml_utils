import numpy as np
import feats

def sum_data(paths):
    part_feats=[feats.read(path_i)[0] for path_i in paths]
    full_feats=feats.Feats()
    for name_i in part_feats[0].keys():
        id_i="%s_1"%name_i  #name_i. subname(3)
        raw_i=[ path_i[id_i] for path_i in part_feats]
        data_i=np.sum(raw_i,axis_i=0)
        raise Exception(data_i.shape)
        full_feats[name_i]=data_i
    return full_feats

true_path="../3DHOI/1D_CNN/feats"
pred_path="student/ae_30_200/feats"
out_path="student/ae_30_200/res"
paths=[,"../conv_frames/student/ae_30_200/feats","../conv_frames/student/ae_30_200/next/feats"]
sum_data(paths)