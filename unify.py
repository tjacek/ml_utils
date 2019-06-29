import numpy as np
import dataset,files

def read(in_path):
    dir_paths=files.top_files(in_path)
    datasets=[dataset.read_dataset(path_i) for path_i in dir_paths]
    unified_dict=datasets[0]
    for dataset_i in datasets[1:]:
        unified_dict=unify_dicts(unified_dict,dataset_i)
    return unified_dict

def unify_dicts(ts_dict1,ts_dict2):
    names=ts_dict1.ts_names()
    new_dict={}
    for name_i in names:
        new_dict[name_i]=unify_ts(ts_dict1[name_i],ts_dict2[name_i])
    return dataset.TSDataset(new_dict,ts_dict1.name)

def unify_ts(ts1,ts2):
    ts_len=min(ts1.shape[0],ts2.shape[0])
    ts1,ts2=ts1[:ts_len,:],ts2[:ts_len,:]
    return np.concatenate([ts1,ts2],axis=1)

if __name__ == "__main__":
    read('mra')
