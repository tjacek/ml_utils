import feats,files
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

def all_plots(in_path,out_path=None,plot_type="cat"):
    if(not out_path):
        out_path='plots_'+in_path
    files.make_dir(out_path)
    for i,path_i in enumerate(files.top_files(in_path)):
        type_i=(plot_type,i)  if(plot_type=="single") else plot_type
        out_i=out_path+'/'+ path_i.split('/')[-1]
        plot_i=tsne_plot(path_i,show=False,plot_type=type_i)  
        plot_i.savefig(out_i,dpi=1000)
        plot_i.close()

def tsne_plot(in_path,show=True,plot_type="cat"):
    feat_dataset= feats.read(in_path) if(type(in_path)==str) else in_path
    feat_dataset=feat_dataset.split()[1]
    tsne=manifold.TSNE(n_components=2,perplexity=30)#init='pca', random_state=0)
    X=tsne.fit_transform(feat_dataset.X)
    y=feat_dataset.get_labels() 
    color_helper=get_colors_helper(feat_dataset.info,plot_type)
    return plot_embedding(X,y,title="tsne",color_helper=color_helper,show=show)

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

def get_colors_helper(info,plot_type="person"):
    if(type(plot_type)==tuple):
        cat_i=plot_type[1]
        def color_helper(i,y_i):
            point_cat=int(info[i].split('_')[0])
            print(point_cat,cat_i)
            return 5*int(point_cat==(cat_i+1))
        return color_helper
    if(plot_type=="cat"):
        return lambda i,y_i: int(info[i].split('_')[0])
    if(plot_type=="full_person"):
        return lambda i,y_i: int(info[i].split('_')[1])
    return lambda i,y_i: int(info[i].split('_')[1]) %2       

if __name__ == "__main__":
#    all_plots('../fusion/feats',"../fusion/plots","single")
    tsne_plot("../MSR_full/feats",show=True,plot_type="cat")