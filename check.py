import numpy as np
import files,dataset
#import exper.inspect,

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

sparse_feats("../old/single_smooth/binary_seq")
