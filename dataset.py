import read
import numpy as np

class Dataset(object):
    def __init__(self,data_array):
        self.size=data_array.shape[0]
        self.dim=data_array.shape[1]
        self.X=data_array

    def get_attr(self,i):
        return self.X[:,i]

    def to_csv(self):
        lines=map(to_csv_line,self.X)
        return array_to_string(lines,sep="\n")

class LabeledDataset(Dataset):
    def __init__(self, data_array,labels):
        super(LabeledDataset, self).__init__(data_array)
        self.y=np.array(labels)
        self.n_cats=max(labels)
        self.cat_names=[str(i) for i in range(self.n_cats)]

    def to_arff(self):
        arff="@RELATION dataset\n"
        atributes=map(get_attr_header,range(self.dim))
        arff+=array_to_string(atributes)
        arff+=get_cats_header(self.cat_names)
        arff+="\n@DATA\n"
        for instance,cat in zip(list(self.X),self.y):
            arff+=to_csv_line(instance)+str(cat)+"\n"
        return arff

    def to_csv(self):
        csv=""
        for instance,label in zip(self.X,self.y):
            csv+=to_csv_line(instance)+"#"+label+"\n"
        return csv

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

def get_attr_header(i):
    return "@ATTRIBUTE attr"+str(i)+" NUMERIC\n"

def get_cats_header(cats):
    cat_header="@ATTRIBUTE class {"
    for cat_i in cats:
        cat_header+=" "+str(cat_i)
    cat_header+="}\n"
    return cat_header

def to_csv_line(array):
    return reduce(lambda x,y,:x+str(y)+",",array,"")

def array_to_string(array,sep=""):
    return reduce(lambda x,y:x+str(y)+sep,array,"")

if __name__ == "__main__":
    path="/home/user/df/exp2/dataset.lb"
    dataset=labeled_to_dataset(path)
    print(dataset.to_arff())

        
