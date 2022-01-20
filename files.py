import os,re#,itertools
from functools import wraps
from collections import defaultdict

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def __len__(self):
        return len(self.split('_'))

    def clean(self):
        digits=[ str(int(digit_i)) 
                for digit_i in re.findall(r'\d+',self)]
        return Name("_".join(digits))

    def get_cat(self):
        return int(self.split('_')[0])-1

    def get_person(self):
        return int(self.split('_')[1])

    def subname(self,k):
        subname_k="_".join(self.split("_")[:k])
        return Name(subname_k)

class NameList(list):
    def __new__(cls, name_list=None):
        if(name_list is None):
            name_list=[]
        return list.__new__(cls,name_list)

    def n_cats(self):
        return len(self.unique_cats())

    def unique_cats(self):
        return set(self.get_cats())

    def get_cats(self):
        return [name_i.get_cat() for name_i in self]     

    def binarize(self,j):
        return [ int(cat_i==0) for cat_i in self.get_cats()]

    def by_cat(self):
        cat_dict={cat_j:NameList() 
                for cat_j in self.unique_cats()}
        for name_i in self:
            cat_dict[name_i.get_cat()].append(name_i)
        return cat_dict

    def cats_stats(self):
        stats_dict={ cat_i:0 for cat_i in self.unique_cats()}
        for cat_i in self.get_cats():
            stats_dict[cat_i]+=1
        return stats_dict

    def subset(self,indexes):
        return NameList([self[i] for i in indexes])

    def filtr(self,cond):
        return NameList([name_i for i,name_i in enumerate(self) 
                           if cond(i,name_i)])

class PathDict(dict):
    def __init__(self, arg=[]):
        super(PathDict, self).__init__(arg)

    def split(self,selector=None):
        train,test=split(self,selector)
        return PathDict(train),PathDict(test)    

def get_path_dict(in_path):
    paths={}
    for path_i in top_files(in_path):
        name_i=get_name(path_i)
        paths[name_i]=top_files(path_i)
    return PathDict(paths)

class SetSelector(object):
    def __init__(self,names):
        self.train=set(names)

    def __call__(self,name_i):
        return name_i in self.train

def get_name(in_path):
    return Name(in_path.split("/")[-1]).clean()

def natural_sort(l):
    return sorted(l,key=natural_keys)

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text)]

def atoi(text):
    return int(text) if text.isdigit() else text

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def dir_function(args=2,recreate=True):
    if(args==2):
        def decor_fun(fun):
            @wraps(fun)
            def dir_decorator(in_path,out_path):
                make_dir(out_path)
                in_iter,out_iter=gen_paths(in_path,out_path)
                output=[]
                for in_i,out_i in zip(in_iter,out_iter):
                    if(recreate or (not os.path.exists(out_i))):
                        output.append(fun(in_i,out_i))
            return dir_decorator          
    else:
        def decor_fun(fun):
            @wraps(fun)
            def dir_decorator(in_path):
                return [fun(path_i) 
                    for path_i in top_files(in_path)]
            return dir_decorator            
    return decor_fun

def gen_paths(in_path,out_path):
    if(type(in_path)==tuple):
        common,binary=in_path
        in_iter=list(zip(top_files(common),top_files(binary)))
        out_iter=[f"{out_path}/{binary_i.split('/')[-1]}"
                for i,(common_i,binary_i) in enumerate(in_iter)]
    else:
        in_iter=top_files(in_path)
        print(in_iter)
        out_iter=[f"{out_path}/{in_i.split('/')[-1]}" 
                   for in_i in in_iter]    
    return in_iter,out_iter

#def get_paths(in_path,name="dtw"):
#    return ["%s/%s" % (path_i,name) 
#                for path_i in top_files(in_path)]

def save_txt(out_path,text):
    if(type(text)==list):
        text="\n".join(text)
    with open(out_path,"w") as out_file:   
        out_file.write(text) 
        out_file.close()

def save(out_path,obj):
    import pickle 
    with open(out_path,"wb") as out_file:   
        pickle.dump(obj,out_file) 
        out_file.close()