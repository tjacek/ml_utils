import numpy as np
import math
import exper,exper.persons,feats,files,learn

def adaptive_votes(votes_path,binary=False,clf_type="SVC",show=True):
    votes=feats.read(votes_path)
    if(binary):
        votes=binarize(votes)
    if(not clf_type):
        train,test=votes.split()
        acc=simple_voting(test)
        print(acc)
        return acc
    return exper.exper_single(votes,clf_type=clf_type,norm=False,show=show)

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
    n_cats=votes.n_cats()
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

def simple_voting(test):
    n_cats=test.n_cats()
    names=test.info
    acc=[]
    for i,x_i in enumerate(test.X):
        size=x_i.shape[0]
        x_i=x_i.reshape((int(size/n_cats),n_cats))
        votes_i=np.sum(x_i,axis=0)
        pred_cat=np.argmax(votes_i)
        true_cat=int(names[i].split('_')[0])-1
        acc.append(int(pred_cat==true_cat))
    return np.mean(acc)

def adaptive_exp(votes_path,out_path=None):
    clf_args=['LR','SVC',"MLP"]
    binary_args=[True,False]
    lines=['clf,prob,accuracy,precision,recall,f1']
    for clf_i in clf_args:
        for binary_j in binary_args:
            y_pred,y_true,names=adaptive_votes(votes_path,binary_j,clf_i,show=False)
            metrics_ij=learn.compute_score(y_true,y_pred,as_str=True)
            line_ij=",".join([clf_i,str(not binary_j),metrics_ij])
            lines.append(line_ij)
    result="\n".join(lines)
    if(not out_path):
        out_path=votes_path+"_result.csv"
    file_str = open(out_path,'w')
    file_str.write(result)
    file_str.close()