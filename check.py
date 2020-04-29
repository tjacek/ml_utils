import numpy as np,os
import files,dataset
import exper.inspect

def sparse_feats(in_path):
    seqs=[ dataset.read_dataset(path_i) 
            for path_i in files.top_files(in_path)]
    feat=[np.sum(np.abs(seq_i.to_array()),axis=0)
            for seq_i in seqs]
    for feat_i in feat:
        threshold(feat_i)

def threshold(feat_i):
    feat_i[feat_i<1]=0.0
    feat_i[ feat_i>1]=1.0
    print(np.sum(feat_i))

def std_threshold(feat_i):
    feat_i=norm(np.sort(feat_i))
    print(feat_i)

def norm(arr):
    arr-=np.mean(arr)
    return arr/np.std(arr)

def ens_inspect(vote_path):
    paths=files.top_files(vote_path)
    for path_i in paths:
        print(path_i)
        print( files.has_dirs(path_i))
        print(ens_stats(path_i))

def ens_stats(path_i):
    acc=exper.inspect.clf_acc(path_i,data="test")
    mean_i=np.mean(acc)
    med_i=np.median(acc)
    max_i,k=np.amax(acc),np.argmax(acc)
    min_i,t=np.amin(acc),np.argmin(acc)
    return "%s,%s,%s,%s,%s,%s"% (max_i,k,mean_i,med_i,min_i,t)

#sparse_feats("../old/single_smooth/binary_seq")
vote_path="proj2/ens3/LR"#/#stats_sim"

ens_inspect(vote_path)
#acc=exper.inspect.clf_acc(votes_path,data="test")
#print(acc)