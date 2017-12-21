import numpy as np

class DataReader(object):
    def __init__(self,parsers=None, data_sep='#'):
        if(parsers==None):
            parsers=[ parse_vector,CatParser(),person_parser]
        self.data_sep=data_sep
        self.parsers=parsers
        
    def __call__(self,out_path):
        raw_file=read_file(out_path)
        basic_data=basic_parse(raw_file,self.data_sep)
        n_samples=len(basic_data)
        n_dim=len(self.parsers)
        def parser_helper(i):
            parser_i=self.parsers[i]
            return [ parser_i(basic_data[j][i])
                        for j in range(n_samples)]
        outputs=[ parser_helper(i)
                   for i in range(n_dim)]
        outputs[0]=np.array(outputs[0])
        return tuple(outputs)

class CatParser(object):
    def __init__(self):
        self.cat2id={}

    def n_cats(self):
        return len(self.cat2id.keys())

    def __call__(self,raw_cat):       
        if(not raw_cat in self.cat2id):
            self.cat2id[raw_cat]=self.n_cats()
        return self.cat2id[raw_cat]

def person_parser(raw_cat):
    return int(raw_cat)

def parse_vector(raw_data):
    return [ float(x_i) 
             for x_i in raw_data.split(',')] 

def basic_parse(lines,data_sep='#'):
    if(type(lines)!=list):
        lines=lines.split('\n')
    return [ line_i.split(data_sep)
             for line_i in lines ] 

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()
    file_object.close()
    return lines