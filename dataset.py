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
        if(data2 is None):
            return data1
        new_X=np.concatenate( [data1.X,data2.X],axis=1)
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
            return { key_j:value_j[i]
                     for key_j,value_j in self.info.items()}
        return [ get_instance(i) 
                 for i in range(len(self))]

    def new_dataset(self,new_X):
        return Dataset(new_X,self.y,self.info)

    def select(self,selector,split=False):
        instances=self.as_instances()
        choosen_insts=[inst_i 
                        for inst_i in instances
                            if(selector(inst_i))]
        if(split):
            clos_selector=lambda x: not selector(x)
            clos_insts=[inst_i 
                        for inst_i in instances
                            if(clos_selector(inst_i))]
            return from_instances( choosen_insts),from_instances(clos_insts)
        return from_instances( choosen_insts) 

def select_single(dataset,i=0):
    def select_helper(inst_i):
        return inst_i['persons']==i
    return dataset.select(select_helper,split=True)
    #pos=lambda inst_i:inst_i['persons']==i
    #neg=lambda inst_i:inst_i['persons']!=i
    #return dataset.select(pos),dataset.select(neg) 

def select_person(dataset,i=0,split=False):
    def person_selector(inst_i):
        return (inst_i['persons'] % 2)==i
    return dataset.select(person_selector,split)         

def select_category(dataset,cats=[]):
    instances=dataset.as_instances()
    if(type(cats)==list):
        cats=Set(cats)  
    s_inst=[ inst_i 
               for inst_i in instances
                 if(int(inst_i['cats']) in cats)]
    return from_instances(s_inst) 

def from_instances(instances):
    x=get_row('X',instances)
    y=get_row('cats',instances)
    basic_attr=Set(['X','cats'])
    attr_names=instances[0].keys()
    info={ name_j:get_row(name_j,instances)
             for name_j in attr_names
               if not (name_j in basic_attr)}
    X=np.array(x)
    return Dataset(X,y,info)

def get_row(key,instances):
    return [ inst[key]
             for inst in instances]

def read_and_unify(data_paths,select=[True,True],norm=[True,True]):
    n_datasets=len(data_paths)
    all_dataset=[ get_dataset(data_path_i)
                  for data_path_i in data_paths]
    def redu_helper(i): #data,select):
        data_i=all_dataset[i]
        select_i=select[i]
        norm_i=norm[i]
        if(norm_i):
            data_i=data_i(preprocessing.scale)
        if(select_i!=False):
            return select_feat.select_feat(data_i,select_i)
        else:
            return data_i
    all_dataset=[redu_helper(i)
                   for i in range(n_datasets)]
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