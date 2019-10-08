import numpy as np
import math
import exper,exper.persons,feats,files

def adaptive_votes(votes_path,binary=False,clf_type="SVC"):
    votes=feats.read(votes_path)
    if(binary):
        votes=binarize(votes)
    exper.exper_single(votes,clf_type=clf_type,norm=True)

def make_votes(args,out_path,clf_type="LR"):
    datasets=exper.voting.get_datasets(**args)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
        train_i,test_i=data_i.split()
        train_votes=exper.persons.pred_by_person(train_i,clf_type)
        test_votes=exper.persons.pred_vectors(train_i,test_i,clf_type)
        votes_dict=dict(train_votes+test_votes)
        votes_feats=feats.from_dict(votes_dict)
        out_i=out_path+'/nn'+str(i)
        votes_feats.save(out_i)

def binarize(votes):
    n_cats=len(set(votes.get_labels()))
    n_clfs=int(votes.X.shape[1]/n_cats)
    X_parts=[votes.X[:,i*n_cats:(i+1)*n_cats]for i in range(n_clfs)]
    binary_X=[np.array([ one_hot(dist_i)
                for dist_i in x_j]) 
                    for x_j in X_parts]
    binary_X=np.concatenate(binary_X,axis=1)
    return feats.FeatureSet(binary_X,votes.info)

def one_hot(dist_i):
    k=np.argmax(dist_i)
    one_hot_i=np.zeros(dist_i.shape)
    one_hot_i[k]=1
    return one_hot_i