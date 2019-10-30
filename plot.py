import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import unify,files,tools,filtr

def muliplot_errors(ts_dataset,errors,out_path):
    names=ts_dataset.ts_names()
    train,test=filtr.split(names)
    by_cat=filtr.by_cat(train)
    feats=get_feats(by_cat,ts_dataset)
    files.make_dir(out_path)
    for i,error_i in enumerate(errors):
        cat_i=filtr.get_cat(error_i)
        error_feats=ts_dataset.as_features(error_i)
        feats_i=feats[cat_i]
        full=[list(cat_j)+[err_j] 
                for cat_j,err_j in zip(feats_i,error_feats)]
        out_i=out_path+'/'+error_i
        files.make_dir(out_i)
        for j,cat_j in enumerate(full):
            colors=['b' for ts_k in cat_j]
            colors[-1]='r'
            out_ij= out_i+'/feats'+str(j)
            save_multi(cat_j,out_ij,colors)

def multiplot(ts_dataset,split=False):
    names=ts_dataset.ts_names()
    names=filtr.split(names)[0] if(split) else filtr.one_per_person(names)
    by_cat=filtr.by_cat(names)
    n_feats=ts_dataset.n_feats()
    feat_paths=dir_struct("multi_"+ts_dataset.name,n_feats)
    colors="bgrcmyk"
    feats=get_feats(by_cat,ts_dataset)
    for i,feats_i in feats.items():
        for j,cats_j in enumerate(feats_i):
            colors=[colors[k % len(colors)] 
                       for k,ts_k in enumerate(cats_j)]
            out_ij=feat_paths[j]+'/cat'+str(i)
            save_multi(cats_j,out_ij,colors)

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
    dir_struct(out_path,ts_dataset.n_feats())
    for name_ts in ts_dataset.ts_names():
        features_i=ts_dataset.as_features(name_ts)
        for j,feat_j in enumerate(features_i):
            path_ij=feats_dict[j]+'/'+name_ts +".png"
            save_ts(feat_j,path_ij)

def dir_struct(out_path,n_feats):
    files.make_dir(out_path)
    feats_dict=[ out_path+'/feat'+str(i) 
                    for i in range(n_feats)]
    for dict_i in feats_dict:
        files.make_dir(dict_i)
    return feats_dict

def get_feats(by_cat,ts_dataset): 
    feats={}
    for i,cat_i in by_cat.items():
        feats_i=[ts_dataset.as_features(name_t)
                    for name_t in cat_i]     
        feats[i]=list(zip(*feats_i))
    return feats

def save_multi(cats_j,out_ij,colors):
    for k,ts_k in enumerate(cats_j):
        plt.plot(ts_k,colors[k])
    print(out_ij)
    plt.savefig(out_ij)
    plt.clf()

def save_ts(ts,out_path):
    if(ts.ndim==2):
        plt.scatter(ts[:,0],ts[:,1]) 
    else:
        x=np.arange(ts.shape[0])
        plt.plot(x,ts)
    plt.savefig(out_path)
    plt.clf()
    plt.close()

def heat_map(conf_matrix,labels=None):
    conf_matrix=np.around(conf_matrix,2)
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}, fmt='g')
    if(labels):
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.show()

if __name__ == "__main__":
    ts_dataset=unify.read("mra")
    heat_map(ts_dataset.feat_corl())