import sys
sys.path.append("..")
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



d=fetch_covtype()
#print(dir(d))
name_list=to_feats(d).names()
print(name_list.cats_stats())