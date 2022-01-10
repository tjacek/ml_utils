import sys
sys.path.append("..")
from functools import wraps
import numpy as np,random
import convert,learn,ens

def bagging_ensemble(dataset,gen ,clf_type="SVC_simple"):
    results=[learn.train_model(dataset)]
    acc=[]
    for data_i in gen:
        result_i=learn.train_model(data_i,clf_type=clf_type)
        acc.append(result_i.get_acc())
        results.append(result_i)
    bag_votes=ens.Votes(results)
    print(acc)
    final_result=bag_votes.voting()
    final_result.report()
    return bag_votes

#def resample_dataset(dataset,n_clf=5):
#    train,test=dataset.split()
#    for i in range(n_clf):
#        sampled_dataset=bagging(train,test)
#        yield sampled_dataset
def gen_data(fun):
    @wraps(fun)
    def gen_decorator(dataset,n_clf=5):
        for i in range(n_clf):
            yield fun(dataset)
    return gen_decorator 

def bagging(train,test=None):
    size=len(train)
    indexes=np.random.randint(0, high=size, size=size)
    names=train.names()
    names=names.subset(indexes)
    print(len(names))
    if(test):
        names=names+ test.names()
    return dataset.subset(names,new_names=True)

@gen_data
def one_out(dataset):
    import clf
    train,test=dataset.split()
    names=train.names()
    y_true,y_pred=[],[]
    for i,name_i in enumerate(names):
        def helper(j,name_j):
            return (i!=j)
        one_out_i=names.filtr(helper)
        data_i=train.subset(one_out_i,new_names=False)
        print(len(data_i))
        data_i=bagging(data_i)
        print(len(data_i))
        model_i=clf.get_cls("LR")
        data_i.norm()
        X,y,_= data_i.as_dataset()
        model_i.fit(X,y)
        y_i=model_i.predict_proba(train[name_i].reshape(1,-1))
        y_true.append(name_i.get_cat())
        y_pred.append(y_i)

    y_pred=np.squeeze(np.array(y_pred))
    return learn.Result(y_true,y_pred,names)
#    for date_i in resample_dataset(dataset,n_clf=5)

def subspace(dataset,alpha=0.75,n_clf=5):
    old_dims= dataset.dim()[0]
    new_dims=int(alpha*old_dims)   
    indexes=list(range(old_dims))
    for i in range(n_clf):
        random.shuffle(indexes)
        indexes=indexes[:new_dims]
        def fun(x):
            new_x=np.array([x[j] for j in indexes])
            return new_x
        yield dataset.transform(fun,copy=True)
 
if __name__ == "__main__":
    dataset=convert.txt_dataset("penglung/raw.data")
#    result=learn.train_model(dataset,clf_type="SVC_simple")
#    result.report()
    results=[result_i for result_i in one_out(dataset,10)]
    final=ens.Votes(results).voting()
    final.report()
#    bag_votes=bagging_ensemble(dataset,gen,clf_type="LR")
#    bag_votes.save("penglung/bag_votes")