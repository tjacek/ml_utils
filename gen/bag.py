import sys
sys.path.append("..")
import numpy as np
import convert,learn,ens

def bagging_ensemble(dataset,n_clf=5):
    results=[]
    for i in range(n_clf):
        data_i=resample_dataset(dataset)
        result_i=learn.train_model(data_i)
        results.append(result_i)
    bag_votes=ens.Votes(results)
    final_result=bag_votes.voting()
    final_result.report()

def resample_dataset(dataset):
    train,test=dataset.split()
    size=len(train)
    indexes=np.random.randint(0, high=size, size=size)
    names=train.names()
    names=names.subset(indexes)
    names+=test.names()
    sampled_dataset=dataset.subset(names)#+test
    return sampled_dataset

if __name__ == "__main__":
    dataset=convert.arff_dataset("wave/raw.arff")
    bagging_ensemble(dataset,n_clf=10)