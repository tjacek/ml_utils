import numpy as np
import matplotlib.pyplot as plt
import exper,exper.cats,filtr,feats
import exper.inspect

def clf_selection(in_path):
    corl_matrix=exper.inspect.all_pred_corl(in_path,None,test=False)
    hard_cats=find_hard_cats(corl_matrix)
    hard_matrix=corl_matrix[:,hard_cats]
    hard_matrix-= np.expand_dims(np.mean(hard_matrix,axis=1),axis=1)
    hard_matrix/=np.expand_dims(np.std(hard_matrix,axis=1),axis=1)
    print(np.argmax(hard_matrix,axis=0))
#    clf_quality=(hard_matrix>1).astype(int)
#    print(np.sum(clf_quality,axis=0))

def find_hard_cats(corl_matrix):
    cat_quality= np.mean(corl_matrix,axis=0)
    print(cat_quality)
    cat_quality-=np.mean(cat_quality)
    cat_quality/=np.std(cat_quality)
    return (cat_quality<-1)

#def clf_selection(in_path,adapt=True):
#    votes=exper.cats.from_binary(in_path)
#    train,test=filtr.split(votes)
#    quality=err_quality(train)  
#    if(adapt):
#        acc=[ adapt_voting(select_votes(k+1,votes,quality))
#                for k in range(len(quality))]
#    else:
#        acc=[ voting(select_votes(k+1,test,quality))
#                for k in range(len(quality))]
#    plot_acc(acc)

#def err_quality(train):
#    n_clfs=train.values()[0].shape[0]
#    names=train.keys()
#    cats=filtr.all_cats(train.keys())
#    wrong_votes={ name_i:np.sum(train[name_i]!=(cats[i]))
#                    for i,name_i in enumerate(names)}
#    def clf_helper(i):
#        quality_i=0
#        for j,name_j in enumerate( names):
#            pred_ij= train[name_j][i] 
#            if(pred_ij==(cats[j])):
#                quality_i+=wrong_votes[name_j]
#        return quality_i
#    return [clf_helper(i) for i in range(n_clfs)]

#def select_votes(k,votes,quality):
#    ind_quality=np.flip(np.argsort(quality))[:k]
#    return { name_i: vote_i[ind_quality] for name_i,vote_i in votes.items()}

#def voting(votes):
#    names=votes.keys()
#    y_true=filtr.all_cats(names)
#    y_pred=[np.argmax(np.bincount(votes[name_i]))  for name_i in names]
#    return np.mean([ int(true_i==pred_i) 
#                        for true_i,pred_i in zip(y_true,y_pred)])

#def adapt_voting(votes):
#    votes=exper.cats.binary_dataset(feats.from_dict(votes))
#    votes.X=np.squeeze(votes.X)
#    return exper.exper_single(votes,clf_type="LR",norm=False)

#def plot_acc(acc):
#    print(acc)    
#    plt.plot(acc)
#    plt.ylabel("acc")
#    plt.show()