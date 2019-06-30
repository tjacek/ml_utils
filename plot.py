import numpy as np
import matplotlib.pyplot as plt
import unify,files,tools

def plot_by_action(ts_dataset):
    out_path=ts_dataset.name+"_actions"
    files.make_dir(out_path)
    for name_ts in ts_dataset.ts_names():
        path_i=out_path+'/'+name_ts
        files.make_dir(path_i)
        features_i=ts_dataset.as_features(name_ts)
        print(name_ts)
        for j,feat_j in enumerate(features_i):
            path_ij=path_i+'/feat'+str(j)+".png"
            save_ts(feat_j,path_ij)

def plot_by_feat(ts_dataset):
    out_path=ts_dataset.name+"_feats"
    files.make_dir(out_path)
    feats_dict=[ out_path+'/feat'+str(i) 
                    for i in range(ts_dataset.n_feats())]
    for dict_i in feats_dict:
        files.make_dir(dict_i)
    for name_ts in ts_dataset.ts_names():
        features_i=ts_dataset.as_features(name_ts)
        for j,feat_j in enumerate(features_i):
            path_ij=feats_dict[j]+'/'+name_ts +".png"
            save_ts(feat_j,path_ij)

def save_ts(ts,out_path):
    if(ts.ndim==2):
        plt.scatter(ts[:,0],ts[:,1]) 
    else:
        x=np.arange(ts.shape[0])
        plt.plot(x,ts)
    plt.savefig(out_path)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    ts_dataset=unify.read("mra")
    ts_dataset=ts_dataset(tools.ResPairs())
    plot_by_feat(ts_dataset)
    #transform=tools.FourrierNoise()
    #ts_dataset=ts_dataset(transform)
    #transform=tools.NormTest()
    #ts_dataset=ts_dataset(transform)
    #x=ts_dataset.to_array()
    #print(np.mean(x,axis=0))