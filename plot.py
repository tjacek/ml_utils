import numpy as np
import matplotlib.pyplot as plt
import read,dataset
#from sets import Set

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
 
def save_ts(ts,out_path):
    x=np.arange(ts.shape[0])
    plt.plot(x,ts)
    plt.savefig(out_path)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    ts_dataset=dataset.read_dataset("seqs/inert")
    plot_ts(ts_dataset)