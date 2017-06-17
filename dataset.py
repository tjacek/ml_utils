import read
import numpy as np
import tools
import select_feat
from sets import Set

class AnnotatedDataset(object):
    def __init__(self,x,y,persons):
        self.X=x
        self.y=y
        self.persons=persons
        print(x.shape)
        #self.dim=x.shape[1]
    
    def dim(self):
        return self.X.shape[1]    

    def get_attr(self,j):
        return self.X[:,j]

    def __getitem__(self,X):
        return self.X[i]

    def __len__(self):
        return len(self.y)
    
    def as_instances(self):
        return [ (self.X[i],self.y[i],self.persons[i]) 
                 for i in range(len(self))]


def select_single(dataset,i=0):
    instances=dataset.as_instances()
    
    pos_inst=[ inst_i 
               for inst_i in instances
                 if(inst_i[2]==i)]
    neg_inst=[ inst_i 
               for inst_i in instances
                 if(inst_i[2]!=i)]
    return from_instances(pos_inst),from_instances(neg_inst)

def select_person(dataset,i=0):
    instances=dataset.as_instances()
    
    s_inst=[ inst_i 
               for inst_i in instances
                 if(inst_i[2] % 2)==i]
    return from_instances(s_inst)          

def select_category(dataset,cats=[]):
    instances=dataset.as_instances()
    cat_set=Set(cats)  
    s_inst=[ inst_i 
               for inst_i in instances
                 if(int(inst_i[1]) in cat_set)]
    return from_instances(s_inst) 

def from_instances(instances):
    x=get_row(0,instances)
    y=get_row(1,instances)
    persons=get_row(2,instances)
    X=np.array(x)
    return AnnotatedDataset(X,y,persons)

def get_row(i,instances):
    return [ inst[i]
             for inst in instances]

def read_and_unify(data_paths,select=[True,True]):
    #data1=get_annotated_dataset(data_path1)
    #data2=get_annotated_dataset(data_path2)
    all_dataset=[ get_annotated_dataset(data_path_i)
                  for data_path_i in data_paths]
    def redu_helper(data,select):
        if(select):
            return select_feat.lasso_model(data)
        else:
            return data
    all_dataset=[redu_helper(data_i,select_i)
                   for data_i,select_i in zip(all_dataset,select)]
    return unify_feat(all_dataset)

def unify_feat(all_dataset):
    all_x=[data_i.X for data_i in all_dataset]
    new_X=np.concatenate(all_x,axis=1)
    
    return AnnotatedDataset(new_X,all_dataset[0].y,all_dataset[0].persons)

def get_annotated_dataset(in_path):
    data_reader=read.DataReader()
    x,y,persons= data_reader(in_path)
    return AnnotatedDataset(x,y,persons)
#class Dataset(object):
#    def __init__(self,data_array,y=None):
#        print(data_array.shape)
#        self.size=data_array.shape[0]
#        self.dim=data_array.shape[1]
#        self.X=data_array
#        self.y=y

#    def get_attr(self,i):
#        return self.X[:,i]

#    def __getitem__(self,with_label=False):
#        if(with_label):
#            return self.X[i],self.y[i]
#        else:
#            return self.X[i]

#    def __len__(self,i):
#        return len(self.y)

#    def __str__(self):
#        lines=[tools.to_csv_line(x_i)
#                 for x_i in self.X] #, range(len(self)) ]
#        return "\n".join(lines)

#class LabeledDataset(Dataset):
#    def __init__(self, data_array,labels):
#        super(LabeledDataset, self).__init__(data_array)
#        self.y=np.array(labels)
#        self.n_cats=max(labels)
#        self.cat_names=[str(i) for i in range(self.n_cats)]
    
#    def binarize(self,k):
#        for i in range(self.size):
#            if(self.y[i]==k):
#                self.y[i]==1.0
#            else:
#                self.y[i]==0.0

#    def __str__(self):
#        csv=""
#        for instance,label in zip(self.X,self.y):
#            csv+=tools.to_csv_line(instance)+",#"+str(label)+"\n"
#        return csv

#class AnnotatedDataset(LabeledDataset):
#    def __init__(self, data_array,labels,annotation):
#         super(AnnotatedDataset, self).__init__(data_array,labels)
#         self.anno=annotation

#    def __str__(self):
#        csv=""
#        for i,instance in enumerate(self.X):
#            postfix=",#"+str(self.y[i])+"#" + str(self.anno[i])+"\n"
#            csv+=tools.to_csv_line(instance)+postfix
#        return csv

#def csv_to_dataset(path):
#    data_list=read.read_csv_file(path)
#    data_array=np.array(data_list)
#    return Dataset(data_array)

#def labeled_to_dataset(path):
#    labeled_list=read.read_labeled(path)
#    data_list=[x[0] for x in labeled_list]
#    labels=[x[1] for x in labeled_list]
#    data_array=np.array(data_list)
#    return LabeledDataset(data_array,labels)

if __name__ == "__main__":
    in_path='../reps/inspect/b_nn/feat.txt'
    data_reader=read.DataReader()
    x,y,persons= data_reader(in_path)
    dataset=AnnotatedDataset(x,y,persons)