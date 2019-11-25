import numpy as np
from sklearn.metrics import accuracy_score

#import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt
#import exper.probs as probs ,files
#import exper,exper.voting,learn
import feats
import exper.cats  

def clf_acc(votes_path):
    votes=feats.read_list(votes_path)
    result=[ pred(vote_i) for vote_i in votes]
    return [accuracy_score(*result_i) for result_i in result]

def pred(data_i):
    y_pred=[np.argmax(x_i) for x_i in data_i.X]
    y_true=data_i.get_labels()
    return y_true,y_pred

def show_votes(votes_path,out_path=None):
    votes=feats.read_list(votes_path)
    full_X=np.array([vote_i.X.T for vote_i in votes]).T
    weights=np.mean(full_X,axis=2)
    votes=feats.FeatureSet(weights,votes[0].info)
    if(out_path):
        votes.save(out_path)
    return accuracy_score(*pred(votes))

def heat_map(conf_matrix,out_path):
    conf_matrix=np.around(conf_matrix,2)
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 4}, fmt='g')
    plt.savefig(out_path,dpi=2000)
    plt.clf()

#def pred_corl(in_path,out_path,test=True):
#    datasets=feats.read_list(in_path)
#    files.make_dir(out_path)
#    for i,data_i in enumerate(datasets):
#        data_i=data_i.split()[int(test)]
#        y_i=to_signal(data_i.get_labels())
#        corl=[[ np.corrcoef(x_j, y_j)[1][0]
#                    for x_j in data_i.X.T]
#                        for y_j in y_i]
#        heat_map(corl,out_path+'/nn'+str(i))

#def all_pred_corl(in_path,out_path,test=True):
#    datasets=feats.read_list(in_path)
#    corl_matrix=[]
#    for i,data_i in enumerate(datasets):
#        data_i=data_i.split()[int(test)]
#        y_i=to_signal(data_i.get_labels())
#        corl=[ np.corrcoef(x_j, y_i[j])[1][0]
#                    for j,x_j in enumerate(data_i.X.T)]
#        corl_matrix.append(corl)            
#    heat_map(np.array(corl_matrix),out_path)

def to_signal(y):
    y=np.array(y)
    n_cats=max(y)
    return [ (y==i+1).astype(float) for i in range(n_cats)]