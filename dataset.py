import read
import numpy as np

class Dataset(object):
    def __init__(self,data_array):
        self.size=data_array.shape[0]
        self.dim=data_array.shape[1]
        self.X=data_array

    def get_attr(self,i):
        return self.X[:,i]

def csv_to_dataset(path):
    data_list=read.read_csv_file(path)
    data_array=np.array(data_list)
    return Dataset(data_array)

if __name__ == "__main__":
    path="/home/user/df/imgs.csv"
    csv_to_dataset(path)
        
