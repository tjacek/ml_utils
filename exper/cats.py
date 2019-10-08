import numpy as np
#from scipy.stats import mode
import exper,exper.persons,feats,files

def adaptive_voting(args,out_path,clf_type="LR"):
    datasets=exper.voting.get_datasets(**args)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
        train_i,test_i=data_i.split()
        train_votes=exper.persons.pred_by_person(train_i)
        test_votes=exper.persons.pred_vectors(train_i,test_i,clf_type)
        votes_dict=dict(train_votes+train_votes)
        votes_feats=feats.from_dict(votes_dict)
        out_i=out_path+'/nn'+str(i)
        votes_feats.save(out_i)
#def adaptive_voting(args,clf_type="LR",out_path=None):    
#    datasets=exper.voting.get_datasets(**args)
#    train_votes=get_train_votes(datasets,clf_type)
#    test_votes=get_test_votes(datasets,clf_type)
#    votes=feats.from_dict( {**train_votes,**test_votes})
#    exper.exper_single(votes,clf_type="LR",norm=False)
#    if(out_path):
#        votes.save(out_path)

#def get_train_votes(datasets,clf_type):
#    votes=[exper.persons.in_sample(data_i,clf_type=clf_type)
#                for data_i in datasets]
#    votes=list(map(list, zip(*votes)))
#    return dict([binary_vector(vote_i) for vote_i in votes])

#def get_test_votes(datasets,clf_type="LR"):
#    votes=[]
#    for data_i in datasets:
#        y_pred,y_true,names=exper.predict_labels(data_i,clf_type)
#        y_pred=np.array(y_pred)-1
#        votes.append([ (name_i,vote_i) for name_i,vote_i in zip(names,y_pred)])
#    votes=list(map(list, zip(*votes)))
#    return dict([binary_vector(vote_i) for vote_i in votes])

#def binary_vector(vote_i):
#    name_i,n_cats=vote_i[0][0],len(vote_i)
#    binary_votes=[]
#    for name_i,cat_j in vote_i:
#        one_hot_j=np.zeros((n_cats,))
#        one_hot_j[cat_j]=1
#        binary_votes.append(one_hot_j)
#    binary_votes=np.concatenate(binary_votes)
#    return name_i,binary_votes