import numpy as np
import exper,exper.persons,feats,files,learn

def adaptive_votes(votes_path,binary=False,clf_type=None,show=True):
    votes=feats.read_list(votes_path)
    print(votes_path)
    if(binary):
        votes=[binarize(vote_i) for vote_i in votes]
    y_pred,y_true,names=voting(votes,clf_type)
    stats=learn.compute_score(y_true,y_pred,as_str=True)
    if(show):
        learn.show_result(y_pred,y_true,names)
        print(learn.compute_score(y_true,y_pred,as_str=True))
    return stats

def make_votes(args,out_path,clf_type="LR",train_data=False):
    datasets=exper.voting.get_data(args)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
        train_i,test_i=data_i.split()
        test_votes=exper.persons.pred_vectors(train_i,test_i,clf_type)#,train_data)        
#        votes_dict=dict(test_votes)
        if(train_data):
            train_votes=exper.persons.pred_by_person(train_i,clf_type)
            votes_dict=dict(train_votes+test_votes)
        else:
            votes_dict=dict(test_votes)
        votes_feats=feats.from_dict(votes_dict)
        out_i=out_path+'/nn'+str(i)
        votes_feats.save(out_i)

def binarize(data_i):
    X=np.array([one_hot(dist_i) 
        for dist_i in data_i.X])
    return feats.FeatureSet(X,data_i.info)

def one_hot(dist_i):
    k=np.argmax(dist_i)
    one_hot_i=np.zeros(dist_i.shape)
    one_hot_i[k]=1
    return one_hot_i

def voting(votes,clf_type=None):
    if(not clf_type):
        return simple_voting(votes)
    else:
        votes=feats.unify(votes)
        return exper.exper_single(votes,clf_type=clf_type,norm=False,show=False)

def simple_voting(votes):
    test=[vote_i.split()[1] for vote_i in votes]
    y_true,names= test[0].get_labels(),test[0].info
    X=np.array([ test_i.X for test_i in test])
    counted_votes=np.sum(X,axis=0)
    y_pred=np.argmax(counted_votes,axis=1)
    return y_pred,y_true,names