import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sets import Set

def unlabeled_plot2D(dataset):
    if(dataset.dim==2):
        fig, ax = plt.subplots()
        x=dataset.get_attr(0)
        y=dataset.get_attr(1)
        ax.scatter(x, y,c='b')
        plt.show()
    else:
        print("Incorect number of dimesion")

def labeled_plot2D(dataset,tabu=[]):
    tabu_set=Set(tabu)
    if(dataset.dim==2):
        fig, ax = plt.subplots()
        x_0=dataset.get_attr(0)
        x_1=dataset.get_attr(1)
        labels=dataset.y
        for i,cat_i in enumerate(labels):
            if(not i in tabu_set):
                cat_i_0=x_0[labels==i]
                cat_i_1=x_1[labels==i]
                color_i=get_color(i)
                shape_i=get_shape(i)
                ax.scatter(cat_i_0,cat_i_1,c=color_i,marker=shape_i)
            else:
                print(i)
        #for i,txt in enumerate(list(labels)):
        #    if(not i in tabu_set):
        #        ax.annotate(str(txt),(x_0[i], x_1[i]))
        plt.show()
    else:
        print("Incorect number of dimesion")

COLORS="bgrcmykw"

def get_color(index):
    i=index % len(COLORS)
    return COLORS[i]

SHAPE='ovs'

def get_shape(index):
    i=index%len(COLORS)
    i=i%len(SHAPE)
    return SHAPE[i]
