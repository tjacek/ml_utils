import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def unlabeled_plot2D(dataset):
    if(dataset.dim==2):
        print("OK")
        fig, ax = plt.subplots()
        x=dataset.get_attr(0)
        y=dataset.get_attr(1)
        ax.scatter(x, y,c='b')
        plt.show()
    else:
        print("Incorect number of dimesion")
