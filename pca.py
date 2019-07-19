import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import feats

def eigen_plot(in_path,n_components=50):
    feat_dataset=feats.read(in_path)
    feat_dataset.norm()
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(feat_dataset.X)
    y=pca.explained_variance_ratio_
    y=np.cumsum(y)
    x=range(n_components)
    print("Explained variance %f" % y[-1])
    plt.bar(x,y)
    plt.show()

eigen_plot("datasets/dtw.txt",200)