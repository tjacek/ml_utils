import read
import numpy as np
import tools
import select_feat
from sets import Set
from sklearn import preprocessing

class Dataset(object):
    def __init__(self,x,y,info):
        self.X=x
        self.y=y
        self.info=info
        self.info['X']=self.X
        self.info['cats']=self.y
    
    def __getitem__(self,i):
        if(type(i)==str):
            return self.info[i]
        return self.X[i]

    def __len__(self):
        return len(self.y)

    def __add__(data1,data2):
        new_X=numpy.concatenate( [data1.X,data2.X])
        return Dataset(new_X,data1.y,data1.info)

    def dim(self):
        return self.X.shape[1]    

    def get_attr(self,j):
        return self.X[:,j]

    def __call__(self,fun):
        new_X=fun(self.X)
        return Dataset(new_X,self.y,self.info)

    def as_instances(self):
        def get_instance(i):
            #x_i=self.X[i]
            #y_i=self.y[i]
            #person_i=self['persons'][i]
            return { key_j:value_j[i]
                     for key_j,value_j in self.info.items()}
        return [ get_instance(i) 
                 for i in range(len(self))]

def select_single(dataset,i=0):
    instances=dataset.as_instances()
    
    pos_inst=[ inst_i 
               for inst_i in instances
                 if(inst_i['persons']==i)]
    neg_inst=[ inst_i 
               for inst_i in instances
                 if(inst_i['persons']!=i)]
    return from_instances(pos_inst),from_instances(neg_inst)

def select_person(dataset,i=0):
    instances=dataset.as_instances()    
    s_inst=[ inst_i 
               for inst_i in instances
                 if(inst_i['persons'] % 2)==i]
    return from_instances(s_inst)          

def select_category(dataset,cats=[]):
    instances=dataset.as_instances()
    cat_set=Set(cats)  
    s_inst=[ inst_i 
               for inst_i in instances
                 if(int(inst_i['cats']) in cat_set)]
    return from_instances(s_inst) 

def from_instances(instances):
    x=get_row('X',instances)
    y=get_row('cats',instances)
    basic_attr=Set(['X','cats'])
    attr_names=instances[0].keys()
    print(attr_names)
    info={ name_j:get_row(name_j,instances)
             for name_j in attr_names
               if not (name_j in basic_attr)}
    X=np.array(x)
    return Dataset(X,y,info)

def get_row(key,instances):
    return [ inst[key]
             for inst in instances]

def read_and_unify(data_paths,select=[True,True],norm=[True,True]):
    #data1=get_annotated_dataset(data_path1)
    #data2=get_annotated_dataset(data_path2)
    n_datasets=len(data_paths)
    all_dataset=[ get_dataset(data_path_i)
                  for data_path_i in data_paths]
    def redu_helper(i): #data,select):
        data_i=all_dataset[i]
        select_i=select[i]
        norm_i=norm[i]
        if(norm_i):
            data_i=data_i(preprocessing.scale)
        if(select_i):
            return select_feat.lasso_model(data_i)
        else:
            return data_i
    all_dataset=[redu_helper(i)
                   for i in range(n_datasets)]#for data_i,select_i in zip(all_dataset,select)]
    return unify_feat(all_dataset)

def unify_feat(all_dataset):
    all_x=[data_i.X for data_i in all_dataset]
    new_X=np.concatenate(all_x,axis=1)
    
    return Dataset(new_X,all_dataset[0].y,all_dataset[0].info)

def get_dataset(in_path):
    data_reader=read.DataReader()
    x,y,persons= data_reader(in_path)
    info={'persons':persons}
    return Dataset(x,y,info)

if __name__ == "__main__":
    in_path='../reps/inspect/b_nn/feat.txt'
    data_reader=read.DataReader()
    x,y,persons= data_reader(in_path)
    dataset=AnnotatedDataset(x,y,persons)