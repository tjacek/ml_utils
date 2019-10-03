import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import exper,exper.voting
import files,feats,filtr

def voting(args,clf_type="LR"):
    votes=votes_dist(args,out_path=None,split=True,clf_type=clf_type)
    vote_dict=as_vote_dict(votes)
    results={name_i:prob_voting(dists_i) for name_i,dists_i in vote_dict.items()}
    show_result(results)

def as_vote_dict(votes):
    votes=[vote_i.to_dict() for vote_i in votes]
    names=list(votes[0].keys())
    vote_dict={}
    for name_i in names:
        vote_dict[name_i]=[vote_i[name_i] for vote_i in votes]
    return vote_dict	

def votes_dist(args,out_path=None,split=True,clf_type="LR"):    
    datasets=exper.voting.get_datasets(**args)
    if(out_path):
        files.make_dir(out_path)
    votes=[]
    for i,data_i in enumerate( datasets):
        model_i=exper.make_model(data_i,clf_type)
        if(split):
            train,test=exper.split_data(data_i)
            X,info=test.X,test.info
        else:
            X,info=data_i.X,data_i.info
        dist_i=model_i.predict_proba(X)
        vote_i=feats.FeatureSet(dist_i,info)
        if(out_path):
            out_i=out_path+'/votes'+str(i)
            vote_i.save(out_i)
        votes.append(vote_i)
    return votes

def show_result(results):
    names=list(results.keys())
    y_true=filtr.all_cats(names)
    y_pred=[  results[name_i] for name_i in names]
    print(classification_report(y_true, y_pred,digits=4))

def prob_voting(votes):
    return np.argmax(np.sum(votes,axis=0)) 

def threshold_voting(votes,thres=0.7):
    s_votes=[vote_i for vote_i in votes
                if(np.amax(vote_i)>thres)]
    if(len(s_votes)==0):
        return simple_voting(votes)
    return simple_voting(s_votes)

def simple_voting(votes):
    cats=np.argmax(votes,axis=1)
    return np.argmax(np.bincount(cats))

def show_kld(args,out_path,clf_type="LR"):
    votes=votes_dist(args,out_path=None,split=True,clf_type=clf_type)
    vote_dict=as_vote_dict(votes)
    files.make_dir(out_path)
    for name_k,vote_i in vote_dict.items():
        kl_matrix=[[ kl_divergence(p, q) for q in vote_i]
                        for p in vote_i]
        out_k=out_path+'/'+name_k
        heat_map(kl_matrix,out_k)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def heat_map(conf_matrix,out_path):
    conf_matrix=np.around(conf_matrix,2)
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 6}, fmt='g')
    plt.savefig(out_path)
    plt.clf()