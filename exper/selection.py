import numpy as np
import exper,exper.voting,exper.persons,exper.cats
import feats,filtr
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def clf_selection(in_path):
    votes=exper.cats.from_binary(in_path)
    train,test=filtr.split(votes)
    quality=err_quality(train)
    acc=[ voting(select_votes(k+1,test,quality))
            for k in range(len(quality))]
    print(acc)

def err_quality(train):
    n_clfs=train.values()[0].shape[0]
    names=train.keys()
    cats=filtr.all_cats(train.keys())
    wrong_votes={ name_i:np.sum(train[name_i]!=(cats[i]))
                    for i,name_i in enumerate(names)}
    def clf_helper(i):
        quality_i=0
        for j,name_j in enumerate( names):
            pred_ij= train[name_j][i] 
            if(pred_ij==(cats[j])):
                quality_i+=wrong_votes[name_j]
        return quality_i
    return [clf_helper(i) for i in range(n_clfs)]

def select_votes(k,votes,quality):
    ind_quality=np.flip(np.argsort(quality))[:k]
    return { name_i: vote_i[ind_quality] for name_i,vote_i in votes.items()}

def voting(votes):
    names=votes.keys()
    y_true=filtr.all_cats(names)
    y_pred=[np.argmax(np.bincount(votes[name_i]))  for name_i in names]
    return np.mean([ int(true_i==pred_i) 
                        for true_i,pred_i in zip(y_true,y_pred)])

#def best_feats(args,clf_type="LR"):
#    datasets=voting.get_datasets(**args)
#    paths=files.top_files(args['deep_paths'])
#    acc=[ exper.persons.pred_acc(data_i,clf_type) for data_i in datasets]
#    return [(paths[i],acc[i]) for i in np.argsort(acc)]

#def acc_curve(dict_args,clf_type="LR"):
#    quality=exper.persons.quality(dict_args,clf_type)
#    votes=get_votes(dict_args,clf_type)
#    acc=[selected_voting(i+1,quality,votes) 
#            for i in range(len(quality)) ]
#    print(acc)    
#    plt.plot(acc)
#    plt.ylabel("acc")
#    plt.show()

#def clf_select(n_clf,args,quality):
#    votes=get_votes(args,clf_type="LR")
#    return selected_voting(n_clf,quality,votes)

#def get_votes(args,clf_type="LR"):
#    datasets=exper.voting.get_datasets(**args)
#    return [exper.predict_labels(data_i,clf_type)  
#                for data_i in datasets]

#def selected_voting(n_clf,quality,votes):
#    print(quality[:n_clf])
#    votes=[votes[i] for i in quality[:n_clf]]
#    y_true,y_pred=exper.voting.count_votes(votes)
#    print(classification_report(y_true, y_pred,digits=4))
#    return accuracy_score(y_true,y_pred)