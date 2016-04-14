import read
import numpy as np
import tools

class Dataset(object):
    def __init__(self,data_array):
        print(data_array.shape)
        self.size=data_array.shape[0]
        self.dim=data_array.shape[1]
        self.X=data_array

    def get_attr(self,i):
        return self.X[:,i]

    def __str__(self):
        lines=map(to_csv_line,self.X)
        return array_to_string(lines,sep="\n")

class LabeledDataset(Dataset):
    def __init__(self, data_array,labels):
        super(LabeledDataset, self).__init__(data_array)
        self.y=np.array(labels)
        self.n_cats=max(labels)
        self.cat_names=[str(i) for i in range(self.n_cats)]
    
    def binarize(self,k):
        for i in range(self.size):
            if(self.y[i]==k):
                self.y[i]==1.0
            else:
                self.y[i]==0.0

    def __str__(self):
        csv=""
        for instance,label in zip(self.X,self.y):
            csv+=tools.to_csv_line(instance)+",#"+str(label)+"\n"
        return csv

class AnnotatedDataset(LabeledDataset):
    def __init__(self, data_array,labels,annotation):
         super(AnnotatedDataset, self).__init__(data_array,labels)
         self.anno=annotation

    def __str__(self):
        csv=""
        for i,instance in enumerate(self.X):
            postfix=",#"+str(self.y[i])+"#" + str(self.anno[i])+"\n"
            csv+=tools.to_csv_line(instance)+postfix
        return csv

def csv_to_dataset(path):
    data_list=read.read_csv_file(path)
    data_array=np.array(data_list)
    return Dataset(data_array)

def labeled_to_dataset(path):
    labeled_list=read.read_labeled(path)
    data_list=[x[0] for x in labeled_list]
    labels=[x[1] for x in labeled_list]
    data_array=np.array(data_list)
    return LabeledDataset(data_array,labels)

def annotated_to_dataset(path):
    labeled_list=read.read_annotated(path)
    data_list=[x[0] for x in labeled_list]
    labels=[x[1] for x in labeled_list]
    annotated=[x[2] for x in labeled_list]
    data_array=np.array(data_list)
    return AnnotatedDataset(data_array,labels,annotated)

if __name__ == "__main__":
    path="/home/user/df/exp2/dataset.lb"
    dataset=labeled_to_dataset(path)
    print(dataset.to_arff())