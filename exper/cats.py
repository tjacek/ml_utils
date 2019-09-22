import numpy as np
from scipy.stats import mode
import exper,exper.persons,feats
#import feats,filtr,exper.persons,exper.voting

def adaptive_voting(args,clf_type="LR",out_path=None):    
    datasets=exper.voting.get_datasets(**args)
    train_votes=get_train_votes(datasets,clf_type)
    test_votes=get_test_votes(datasets,clf_type)
    votes=feats.from_dict( {**train_votes,**test_votes})
    exper.exper_single(votes,clf_type="LR",norm=False)
    if(out_path):
        votes.save(out_path)

def get_train_votes(datasets,clf_type):
    votes=[exper.persons.in_sample(data_i,clf_type=clf_type)
                for data_i in datasets]
    votes=list(map(list, zip(*votes)))
    return dict([binary_vector(vote_i) for vote_i in votes])

def get_test_votes(datasets,clf_type="LR"):
    votes=[]
    for data_i in datasets:
        y_pred,y_true,names=exper.predict_labels(data_i,clf_type)
        y_pred=np.array(y_pred)-1
        votes.append([ (name_i,vote_i) for name_i,vote_i in zip(names,y_pred)])
    votes=list(map(list, zip(*votes)))
    return dict([binary_vector(vote_i) for vote_i in votes])

def binary_vector(vote_i):
    name_i,n_cats=vote_i[0][0],len(vote_i)
    binary_votes=[]
    for name_i,cat_j in vote_i:
        one_hot_j=np.zeros((n_cats,))
        one_hot_j[cat_j]=1
        binary_votes.append(one_hot_j)
    binary_votes=np.concatenate(binary_votes)
    return name_i,binary_votes

#def make_cat_feats(args,out_path,clf_type="LR",binary=True):
#    datasets=exper.voting.get_datasets(**args)
#    pred_dicts=[pred_dict(data_i,clf_type) 
#                    for data_i in datasets]
#    names=pred_dicts[0].keys()
#    names.sort()
#    def cat_helper(name_i):
#        return np.array([pred_j[name_i] for pred_j in pred_dicts])
#    pred_feats={name_i:cat_helper(name_i) for name_i in names}
#    pred_feats=feats.from_dict(pred_feats)
#    if(binary):
#    	pred_feats=binary_dataset(pred_feats)
#    pred_feats.save(out_path)

#def binary_dataset(data_i):
#    n_cats=int(np.amax(data_i.X))
#    def one_hot(cat_j):
#        one_hot=np.zeros((n_cats+1,))
#        one_hot[int(cat_j)]=1
#        return one_hot
#    new_X=[np.concatenate([one_hot(cat_j) 
#                            for cat_j in x_i])
#    	            for x_i in data_i.X]
#    return feats.FeatureSet(np.array(new_X),data_i.info)

#def from_binary(in_path):
#    binary_data=feats.read(in_path)
#    n_cats= filtr.n_cats(binary_data.info)
#    n_clfs=binary_data.dim()/n_cats
#    def binary_helper(x_i):
#        one_hot=[ x_i[j*n_cats:(j+1)*n_cats] for j in range(n_clfs)]
#        return np.array([ np.argmax(vec_j) for vec_j in one_hot])
#    binary_data=binary_data.to_dict()
#    return { name_i:binary_helper(x_i) 
#                for name_i,x_i in binary_data.items()}

#def error_votes(votes):
#    names=votes.keys()
#    y_true=filtr.all_cats(names)
#    y_pred=[mode(votes[name_i]).mode[0]+1 for name_i in names]
#    err_names=filtr.erros(names,y_true,y_pred)
#    return filtr.filtered_dict(err_names,votes)