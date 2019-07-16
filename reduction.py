import feats
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

def show_feats(in_path):
    feat_dataset=feats.read(in_path)
    tsne=manifold.TSNE(n_components=2,perplexity=30)#init='pca', random_state=0)
    X=tsne.fit_transform(feat_dataset.X)
    y=feat_dataset.get_labels()
    #color_helper=lambda i,y_i:y_i
    color_helper=PersonColors(feat_dataset)
    plot_embedding(X,y,title="tsne",color_helper=color_helper,show=True)

def plot_embedding(X,y,title="plot",color_helper=None,show=True):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i,y_i:0

    plt.figure()
    ax = plt.subplot(111)

    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})
    print(x_min,x_max)
    #plt.xticks(np.arange(x_min, x_max, 0.005)), plt.yticks([])
    if title is not None:
        plt.title(title)
    if(show):
        plt.show()
    return plt

class PersonColors(object):
    def __init__(self, dataset):
        self.info=dataset.info

    def __call__(self,i,y_i):
        person_i=int(self.info[i].split('_')[1])
        return person_i %2       

if __name__ == "__main__":
    show_feats('datasets/noise.txt')