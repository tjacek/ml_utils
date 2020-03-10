import numpy as np
from sklearn.metrics import accuracy_score
import feats,learn
import exper.persons,exper.selection,exper.curve
import exper.cats
from exper.voting import get_datasets

def acc_csv(in_path,out_path):        
    exper.curve.acc_to_csv(in_path,out_path,get_acc)

def make_plots(in_path,out_path):        
    exper.curve.all_curves(in_path,out_path,get_acc)

def get_acc(path_i):
    clf_ord=orth_selection(path_i)
    results=exper.selection.selected_voting(path_i,clf_ord)
    return [accuracy_score(result_i[0],result_i[1]) 
                for result_i in results]

def orth_selection(in_path):
    votes=feats.read_list(in_path)
    train=[ vote_i.split()[0] for vote_i in votes]
    acc_i=[person_acc(train_i) for train_i in train]
    print(acc_i)
    clf_ord=np.argsort(acc_i)
    print(clf_ord)
    return clf_ord

def person_acc(train_i):
    person_i=[ int(info_j.split("_")[1])
                for info_j in train_i.info]
    clf_i=learn.get_cls("LR")
    clf_i.fit(train_i.X,person_i)
    person_predict=clf_i.predict(train_i.X)
    return accuracy_score(person_i,person_predict)

def full_exper(hc_path,deep_paths,n_feats=500):
    deep_data=get_datasets(hc_path,deep_paths,n_feats)
    deep_train=[ data_i.split()[0] for data_i in deep_data]
    acc=[person_acc(train_i) for train_i in deep_train]
#    print(acc)
    clf_ord=np.argsort(acc)
#    print(clf_ord)
    full_data=get_datasets(hc_path,deep_paths,n_feats)
    prob=get_cat_dist(full_data)
    results=exper.selection.selected_voting(prob,clf_ord)
    true_acc=[accuracy_score(result_i[0],result_i[1]) for result_i in results]
    print(true_acc)

def get_cat_dist(full_data):    
    full_data=[data_i.split() for data_i in full_data]
    prob=[exper.persons.pred_vectors(train_j,test_j,"LR")
            for train_j,test_j in full_data]
    return[ feats.from_dict(dict(prob_i)) for prob_i in prob]

#    results=[exper.predict_labels(data_i,"LR") for data_i in full_data]
#    true_acc=[accuracy_score(result_i[0],result_i[1]) for result_i in results]
#    print(true_acc)