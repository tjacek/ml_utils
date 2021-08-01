import numpy as np
import feats,learn,files

def sum_data(paths):
    part_feats=[feats.read(path_i)[0] for path_i in paths]
    full_feats=feats.Feats()
    for name_i in part_feats[0].keys():
        raw_i=[ path_i[name_i] for path_i in part_feats]
        data_i=np.sum(raw_i,axis=0)
        full_feats[name_i]=data_i
    return full_feats

def ens_acc(in_path):
    import os.path
    results=[]
    for path_i in files.top_files(in_path):
        in_i="%s/feats" % path_i
        if(os.path.isfile(in_i)): 
            result_i=learn.train_model(in_i)#feats.read(in_i)[0]
            print(result_i.get_acc())
            results.append(result_i)
    return results

def find_best(results):
    acc=[ result_i.get_acc() for result_i in results]
    k=np.argmax(acc)
    return results[k]

in_path="../3DHOI2/early/"
results=ens_acc(in_path)
result_i=find_best(results)
print(result_i.get_cf())