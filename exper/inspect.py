import numpy as np
import pandas as pd
import seaborn as sn
import feats
import matplotlib.pyplot as plt
import exper.probs as probs ,files

def show_errors(args,out_path,clf_type="LR"):
    def helper(name_i,votes):
        true_cat=files.natural_keys(name_i)[1]-1
        pred_cat=probs.simple_voting(votes)
        return true_cat!=pred_cat
    show_probs(args,out_path,clf_type="LR",selector=helper)

def show_probs(args,out_path,clf_type="LR",selector=None):
    votes=probs.votes_dist(args,out_path=None,split=True,clf_type=clf_type)
    vote_dict=probs.as_vote_dict(votes)
    files.make_dir(out_path)
    if(not selector):
        selector=lambda x,y:True
    for name_k,vote_i in vote_dict.items():
        if(selector(name_k,vote_i)):
            kl_matrix=np.array(vote_i)
            out_k=out_path+'/'+name_k
            heat_map(kl_matrix,out_k)

def heat_map(conf_matrix,out_path):
    conf_matrix=np.around(conf_matrix,2)
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 4}, fmt='g')
    plt.savefig(out_path,dpi=2000)
    plt.clf()

def pred_corl(in_path,out_path,test=True):
    datasets=feats.read_list(in_path)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
        data_i=data_i.split()[int(test)]
        y_i=to_signal(data_i.get_labels())
        corl=[[ np.corrcoef(x_j, y_j)[1][0]
                    for x_j in data_i.X.T]
                        for y_j in y_i]
        heat_map(corl,out_path+'/nn'+str(i))

def to_signal(y):
    y=np.array(y)
    n_cats=max(y)
    return [ (y==i+1).astype(float) for i in range(n_cats)]