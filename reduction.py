import dataset
import plot
from sklearn import manifold
import sklearn
import sys
import numpy as np

def tsne_reduction(data,dim=2,change=False):
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    X_prim=tsne.fit_transform(data.X)
    if(change):
        data.X=X_prim
        return data
    return X_prim

def lle_reduction(data,dim=2,n_neighbors=20):
    lle= manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='standard')
    X_prim=lle.fit_transform(data.X)
    return X_prim

def hessian_reduction(data,dim=2,n_neighbors=15):
    lle= manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='hessian',eigen_solver="dense")
    X_prim=lle.fit_transform(data.X)
    return X_prim

def spectral_reduction(data,dim=2):
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
                                     eigen_solver="arpack")
    X_prim=embedder.fit_transform(data.X)
    return X_prim

def pca_reduction(data,dim=2,change=False):
    embedder = sklearn.decomposition.SparsePCA(n_components=dim)
    X_prim=embedder.fit_transform(data.X)
    if(change):
        data.X=X_prim
        return data
    return X_prim

def no_reduction(data,dim=2):
    return data.X

reductions=[no_reduction,tsne_reduction,lle_reduction,hessian_reduction,spectral_reduction,pca_reduction]

def show_unlabeled(path,reduction_id):
    data=dataset.csv_to_dataset(path)
    tsne_X=reductions[reduction_id](data)
    tsne_data=dataset.Dataset(tsne_X)
    plot.unlabeled_plot2D(tsne_data)

def show_labeled(path,reduction_id,tabu=[]):
    data=  dataset.get_annotated_dataset(path)  #dataset.labeled_to_dataset(path)
    tsne_X= tsne_reduction(data) #reductions[reduction_id](data)
    tsne_data= dataset.AnnotatedDataset(tsne_X,data.y,data.persons)
    print(tsne_data.dim)
    plot.labeled_plot2D(tsne_data,tabu)

def parse_args(args):
    if(len(sys.argv)==1):
        reduction_id=0
    else:    
        reduction_id=int(sys.argv[1])
    return reduction_id

if __name__ == "__main__":
    in_path='../reps/inspect/b_nn/feat.txt'
    show_labeled(in_path,0)#,[5,6])

