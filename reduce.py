import dataset
import plot
from sklearn import manifold
import sys

def tsne_reduction(data,dim=2):
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    X_prim=tsne.fit_transform(data.X)
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

reductions=[tsne_reduction,lle_reduction,hessian_reduction,spectral_reduction]

def show_unlabeled(path):
    data=dataset.csv_to_dataset(path)
    tsne_X=tsne_reduction(data)
    tsne_data=dataset.Dataset(tsne_X)
    plot.unlabeled_plot2D(tsne_data)

def show_labeled(path,reduction_id):
    data=dataset.annotated_to_dataset(path)
    tsne_X=reductions[reduction_id](data)
    tsne_data=dataset.LabeledDataset(tsne_X,data.y)
    plot.labeled_plot2D(tsne_data)

if __name__ == "__main__":
    path="../af/cascade4/full_dataset"
    if(len(sys.argv)==1):
        reduction_id=0
    else:    
        reduction_id=int(sys.argv[1])
    show_labeled(path,reduction_id)

