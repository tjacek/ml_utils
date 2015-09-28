import read
import numpy as np

class Dataset(object):
    def __init__(self,data_array):
        self.size=data_array.shape[0]
        self.dim=data_array.shape[1]
        self.X=data_array

    def get_attr(self,i):
        return self.X[:,i]

class LabeledDataset(Dataset):
    def __init__(self, data_array,labels):
        super(LabeledDataset, self).__init__(data_array)
        self.labels=labels

def csv_to_dataset(path):
    data_list=read.read_csv_file(path)
    data_array=np.array(data_list)
    return Dataset(data_array)

def labeled_to_dataset(path):
    labeled_list=read.read_labeled(path)
    data_list=map(lambda x:x[0],labeled_list)
    labels=map(lambda x:x[1],labeled_list)
    data_array=np.array(data_list)
    return LabeledDataset(data_array,labels)

if __name__ == "__main__":
    path="/home/user/df/exp2/dataset.lb"
    dataset=labeled_to_dataset(path)
    print(dataset.size)
    print(dataset.dim)
        
