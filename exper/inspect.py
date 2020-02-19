import numpy as np
from sklearn.metrics import accuracy_score

import feats
import exper.cats  

def clf_acc(votes_path,data="test"):
    votes=feats.read_list(votes_path)
    if(data=="test"):
        votes=[ vote_i.split()[1] for vote_i in votes]
    if(data=="train"):
        votes=[ vote_i.split()[0] for vote_i in votes]
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

#def heat_map(conf_matrix,out_path):
#    conf_matrix=np.around(conf_matrix,2)
#    dim=conf_matrix.shape
#    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
#    sn.set(font_scale=1.0)#for label size
#    sn.heatmap(df_cm, annot=True,annot_kws={"size": 4}, fmt='g')
#    plt.savefig(out_path,dpi=2000)
#    plt.clf()

def to_signal(y):
    y=np.array(y)
    n_cats=max(y)
    return [ (y==i+1).astype(float) for i in range(n_cats)]