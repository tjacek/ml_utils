import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import files
from exper.probs import votes_dist,as_vote_dict

def show_probs(args,out_path,clf_type="LR"):
    votes=votes_dist(args,out_path=None,split=True,clf_type=clf_type)
    vote_dict=as_vote_dict(votes)
    files.make_dir(out_path)
    for name_k,vote_i in vote_dict.items():
        kl_matrix=np.array(vote_i)
        out_k=out_path+'/'+name_k
        heat_map(kl_matrix,out_k)

def heat_map(conf_matrix,out_path):
    conf_matrix=np.around(conf_matrix,2)
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 6}, fmt='g')
    plt.savefig(out_path)
    plt.clf()