import files

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)
    
    def n_cats(self):
        return self.names().n_cats()

    def names(self):
        keys=sorted(self.keys(),key=files.natural_keys) 
        return files.NameList(keys)

    def split(self,selector=None):
        train,test=split(self,selector)
        return self.__class__(train),self.__class__(test)

    def transform(self,fun,copy=False):
        new_dict= self.__class__() if(copy) else self
        for name_i,data_i in self.items():
            new_dict[name_i]=fun(data_i)
        return new_dict

    def subset(self,names,new_names=False):
        sub_dict=self.__class__()
        for i,name_i in enumerate( names):
            value_i=self[name_i]
            if(new_names):
                name_i=f'{name_i}_{i}'
            sub_dict[name_i]=value_i
        return sub_dict 

def split(data_dict,selector=None,pairs=True):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in data_dict.keys():
        pair_i=(name_i,data_dict[name_i]) if(pairs) else name_i
        if(selector(name_i)):
            train.append(pair_i)
        else:
            test.append(pair_i)
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1