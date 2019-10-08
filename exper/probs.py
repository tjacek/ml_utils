import numpy as np
from sklearn.metrics import classification_report
import exper,exper.voting
import files,feats,filtr

def voting(args,clf_type="LR",voting_type="simple"):
    votes=votes_dist(args,out_path=None,split=True,clf_type=clf_type)
    vote_dict=as_vote_dict(votes)
    voting_system=get_voting_system(voting_type)
    results={name_i:voting_system(dists_i) for name_i,dists_i in vote_dict.items()}
    show_result(results)

def get_voting_system(voting_type):
    if(voting_type=="prob"):
        return prob_voting
    return simple_voting

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

def kld_voting(votes):
    kld_matrix=get_kld_matrix(votes)
    i=np.argmax(np.sum(kld_matrix,axis=0)) 
    return np.argmax(votes[i])

def get_kld_matrix(vote_i):
	return np.array([[ kl_divergence(p, q) 
	            for q in vote_i]
                    for p in vote_i])

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def max_voting(votes):
    votes=np.array(votes)
    return np.argmax(votes) % votes.shape[0]	
