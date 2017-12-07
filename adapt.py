from sets import Set
import exper,dataset,eval,voting,learn

class EnsembleDataset(object):
    def __init__(self, basic_feats,adapt_feats,
                    norm=True,filter_data=None):
        if(type(basic_feats)==int):
            basic_feats=('rfe',n_feats)
        if(type(adapt_feats)==int):
            adapt_feats=('rfe',n_feats)
        self.basic_feats=basic_feats
        self.adapt_feats=adapt_feats
        self.norm=norm
        self.filter_data=filter_data

    def __call__(self,basic_paths,adapt_paths):
        basic_data=self.basic_dataset(basic_paths)
        if(len(adapt_paths)==0):
            return [basic_data]
        ensemble_datasets=self.adapt_datasets(adapt_paths)
        return [ data_i + basic_data 
                for data_i in ensemble_datasets]

    def basic_dataset(self,paths):
        if(paths is None):
            return None
        data=exper.single_dataset(paths)
        data=exper.feat_selection(data,self.basic_feats,norm=True)
        if(self.filter_data!=None):
            data=self.filter_data(data)
        return data

    def adapt_datasets(self,paths):  
        data=[self.adapt_data(path_i)
                for path_i in zip(paths)]
        if(self.filter_data!=None):
            data=[self.filter(data_i) 
                    for data_i in data]
        return data        

    def adapt_data(self,path_i):
        path_i=get_list_of_paths(path_i)
        data=exper.single_dataset(path_i)
        old_y=data.y
        data=exper.feat_selection(data,self.adapt_feats,norm=self.norm)
        data.y=old_y
        return data           

class SetFilter(object):
    def __init__(self,allowed):
        self.alowed=Set(allowed)

    def __call__(self,data):
        return dataset.select_category(data,self.alowed) 

def get_list_of_paths(path_i):
    print(path_i)
    if(type(path_i)==tuple):
        path_i=list(path_i)
    if(type(path_i)!=list):
        path_i=[path_i]
    return path_i

def gen_pahts(nn_path,indices):
    if(type(indices)==int):
        indices=range(indices)
    if(type(indices)==tuple):
        indices=range(indices[0], indices[1])
    return [nn_path+str(i+1)
                    for i in indices]

if __name__ == "__main__":
    basic_paths="conf/no_deep.txt"  
    nn_path='../AArtyk/all_feats/nn_' 
    basic_paths=['../AArtyk/simple/all/simple.txt',
                 '../AArtyk/simple/corl/dtw_dataset.txt',
                 '../AArtyk/simple/max_z/dtw_dataset.txt']
    #nn_path='../AArtyk3/all_feats/nn_'
    #basic_paths=['../AArtyk2/basic/simple/simple.txt',
    #             '../AArtyk2/basic/corel/dtw_feats.txt',
    #            '../AArtyk2/basic/extr/dtw_feats.txt']
    adapt_paths=[nn_path+str(i)
                    for i in range(20)]
    ensemble_dataset=EnsembleDataset(('rfe',150),('rfe',50),
                                    filter_data=SetFilter([1,2,4,5,9,12,17,19]))
    datasets=ensemble_dataset(basic_paths,adapt_paths)
    print(datasets)
    ensemble=voting.get_ensemble('lr') 
    ensemble_pred,y_true=ensemble(datasets)
    learn.show_result(ensemble_pred,y_true,conf=True)