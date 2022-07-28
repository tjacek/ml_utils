import os,numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import exp,feats,files,seqs

@files.dir_function()
@files.dir_function()
def make_plots(in_path,out_path):
    d=seqs.read_data(in_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    x=np.arange(d.shape[0])
    d=normalize(d)
    for ts_i in d.T:
#        ts_i-=np.mean(ts_i)
#        ts_i/=np.std(ts_i)
        ax.plot(x, ts_i)
        print(d.shape[0])
        print(ts_i.shape)   
    plt.title(out_path.split('/')[-1])
    plt.show()

def normalize(ts):
    for ts_i in ts.T:
        ts_i-=np.mean(ts_i)
        ts_i/=np.std(ts_i)
    return ts

def tsne_plot(in_path,show=True,color_helper="cat",names=False,data="full"):
    feat_dataset= feats.read(in_path)[0]
    if(data=="train"):
        feat_dataset=feat_dataset.split()[0]
    elif(data=="test"):
        feat_dataset=feat_dataset.split()[1]
    tsne=manifold.TSNE(n_components=2,perplexity=30)
    X,y,names=feat_dataset.as_dataset()
    X=tsne.fit_transform(X)
    color_helper=lambda i,y_i:y_i 
    return plot_embedding(X,y,title="tsne",color_helper=color_helper,show=show,names=names)

def plot_embedding(X,y,title="plot",color_helper=None,show=True,names=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i,y_i:0

    plt.figure()
    ax = plt.subplot(111)

    rep= names if(names) else y
    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1],str(rep[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})
    print(x_min,x_max)
    if title is not None:
        plt.title(title)
    if(show):
        plt.show()
    return plt

in_path="../CZU-MHAD/test_spline"
make_plots(in_path,"test")