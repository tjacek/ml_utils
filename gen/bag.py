import sys
sys.path.append("..")
import numpy as np,random
import convert,learn,ens

def bagging_ensemble(dataset,n_clf=5,clf_type="SVC_simple"):
    results=[]#learn.train_model(dataset)]
    acc=[]
    for i in range(n_clf):
        data_i=resample_dataset(dataset)
#        data_i=subspace(dataset,alpha=0.75)
        result_i=learn.train_model(data_i,clf_type=clf_type)
        acc.append(result_i.get_acc())
        results.append(result_i)
    bag_votes=ens.Votes(results)
    print(acc)
    final_result=bag_votes.voting()
    final_result.report()
    return bag_votes

def resample_dataset(dataset):
    train,test=dataset.split()
    size=len(train)
    indexes=np.random.randint(0, high=size, size=size)
    names=train.names()
    names=names.subset(indexes)
    names+=test.names()
    sampled_dataset=dataset.subset(names,new_names=False)#+test
    return sampled_dataset

def subspace(dataset,alpha=0.75):
    old_dims= dataset.dim()[0]
    new_dims=int(alpha*old_dims)
    indexes=list(range(old_dims))
    random.shuffle(indexes)
    indexes=indexes[:new_dims]
    def fun(x):
        new_x=np.array([x[j] for j in indexes])
        return new_x
    return dataset.transform(fun,copy=True)
 
if __name__ == "__main__":
    dataset=convert.txt_dataset("penglung/raw.data")
#    result=learn.train_model(dataset,clf_type="SVC_simple")
#    result.report()
    bag_votes=bagging_ensemble(dataset,n_clf=10,clf_type="LR")
    bag_votes.save("penglung/bag_votes")