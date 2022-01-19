import files

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)

    def names(self):
        keys=sorted(self.keys(),key=files.natural_keys) 
        return files.NameList(keys)

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return self.__init__(train),self.__init__(test)

    def transform(self,fun,copy=False):
        new_dict= self.__init__() if(copy) else self
        for name_i,data_i in self.items():
            new_dict[name_i]=fun(data_i)
        return new_dict

def split(data_dict,selector=None,pairs=True):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in data_dict.keys():
        pair_i=(name_i,dict[name_i]) if(pairs) else name_i
        if(selector(name_i)):
            train.append(pair_i)
        else:
            test.append(pair_i)
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1