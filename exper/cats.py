import numpy as np
import exper,exper.persons,feats,files,learn
import matplotlib.pyplot as plt

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
        test_votes=exper.persons.pred_vectors(train_i,test_i,clf_type)
        if(train_data):
             raise Exception("no train_data")
#            train_votes=exper.persons.pred_by_person(train_i,clf_type)
#            votes_dict=dict(train_votes+test_votes)
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

#def adaptive_exp(votes_path,out_path=None):
#    clf_args=['LR','SVC',"MLP"]
#    binary_args=[True,False]
#    lines=['clf,prob,accuracy,precision,recall,f1']
#    for clf_i in clf_args:
#        for binary_j in binary_args:
#            y_pred,y_true,names=adaptive_votes(votes_path,binary_j,clf_i,show=False)
#            metrics_ij=learn.compute_score(y_true,y_pred,as_str=True)
#            line_ij=",".join([clf_i,str(not binary_j),metrics_ij])
#            lines.append(line_ij)
#    result="\n".join(lines)
#    if(not out_path):
#        out_path=votes_path+"_result.csv"
#    file_str = open(out_path,'w')
#    file_str.write(result)
#    file_str.close()

def acc_curve(vote_path,ord,binary=False):
    votes=selected_votes(vote_path,ord,binary)
    n_clf=len(votes)
    results=[voting(votes[:(i+1)],None) for i in range(n_clf)]
    acc=[ learn.compute_score(result_i[0],result_i[1],False)[0] 
            for result_i in results]
    show_curve(acc,len(results),vote_path,binary)
    return acc

def selected_votes(vote_path,ord,binary=False):
    votes=feats.read_list(vote_path)
    if(binary):
        votes=[binarize(vote_i) for vote_i in votes]
    return [ votes[k] for k in ord]
    
def show_curve(acc,n_clf,vote_path,binary):
    title="_".join(vote_path.split('/'))
    title=title.replace(".._","").replace("votes","")
    voting_type= "HARD" if(binary) else "SOFT"
    title+=voting_type
    plt.title(title)
    plt.grid(True)
    plt.xlabel('number of classifiers')
    plt.ylabel('accuracy')
    plt.plot(range(1,n_clf+1), acc, color='red')
    plt.show()
    return acc