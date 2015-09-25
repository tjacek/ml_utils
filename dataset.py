import io
import numpy as np

class Dataset(object):
    def __init__(self,data_array):
        self.n_instances=data_array.shape[0]
        self.n_features=data_array.shape[1]
        self.X=data_array

def csv_to_dataset(path):
    data_list=io.read_csv_file(path)
    data_array=np.array(data_list)
    return Dataset(data_array)

if __name__ == "__main__":
    path="/home/user/df/imgs.cvs"
    csv_to_dataset(path)
        
