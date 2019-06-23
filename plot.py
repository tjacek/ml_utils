import numpy as np
import matplotlib.pyplot as plt
import read,dataset,tools

def plot_ts(ts_dataset):
    out_path=ts_dataset.name
    read.make_dir(out_path)
    for name_ts in ts_dataset.ts_names():
        path_i=out_path+'/'+name_ts
        read.make_dir(path_i)
        features_i=ts_dataset.as_features(name_ts)
        print(name_ts)
        for j,feat_j in enumerate(features_i):
            path_ij=path_i+'/feat'+str(j)+".png"
            save_ts(feat_j,path_ij)

def plot_by_feat(ts_dataset):
    out_path=ts_dataset.name+"_feats"
    read.make_dir(out_path)
    feats_dict=[ out_path+'/feat'+str(i) 
                    for i in range(ts_dataset.n_feats())]
    for dict_i in feats_dict:
        read.make_dir(dict_i)
    for name_ts in ts_dataset.ts_names():
        features_i=ts_dataset.as_features(name_ts)
        for j,feat_j in enumerate(features_i):
            path_ij=feats_dict[j]+'/'+name_ts +".png"
            save_ts(feat_j,path_ij)

def save_ts(ts,out_path):
    x=np.arange(ts.shape[0])
    plt.plot(x,ts)
    plt.savefig(out_path)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    ts_dataset=dataset.read_dataset("seqs/inert")
    transform=tools.Fourrier()
    ts_dataset=ts_dataset(transform)
    plot_by_feat(ts_dataset)