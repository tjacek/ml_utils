import sys
sys.path.append("..")
import numpy as np
from sklearn.datasets import fetch_covtype
import feats,files

def to_feats(d):
    data_pairs=zip(d['data'],d['target'])
    data_dict=feats.Feats()
    for i,(x_i,y_i) in enumerate(data_pairs):
        cat_i=(i %2)
        name_i=files.Name(f'{y_i}_{cat_i}_{i}')
        data_dict[name_i]=x_i
    return data_dict

def forest_dataset( fraction=0.6):
    d=fetch_covtype()
    forest_feats=to_feats(d)
    name_list=forest_feats.names()
    cats_stats= name_list.cats_stats()
    min_cat=min(cats_stats, key=cats_stats.get)
    cat_size=int(fraction* cats_stats[min_cat])
    by_cat =name_list.by_cat()
    train=balanced_dataset(by_cat,cat_size)
    def helper(name_i):
        cat_i,_,i=name_i.split('_')
        return f'{cat_i}_{int(name_i in train)}_{i}'
    rename_dict={name_i:helper(name_i) for name_i in forest_feats.keys()}
    return forest_feats.rename(rename_dict)

def balanced_dataset(by_cat,size):
    import random
    train=[]
    for cat_j in by_cat.values():
        random.shuffle(cat_j)
        train+=cat_j[:size]
    return set(train)

def arff_dataset(in_path):
    from scipy.io import arff
    dataset = arff.loadarff(in_path)
    feat_dict=feats.Feats()
    for i,data_i in enumerate(dataset[0]):
        data_i=list(data_i)
        x_i=np.array(data_i[:-1])
        cat_i=int(data_i[-1])+1
        name_i=f"{cat_i}_{i%2}_{i}"
        feat_dict[name_i]=x_i
    return feat_dict

if __name__ == "__main__":
#    data_i=forest_dataset()
     data=arff_dataset("wave/raw.arff")
     print(len(data))
#    import learn
#    result=learn.train_model(data)
#    result.report()