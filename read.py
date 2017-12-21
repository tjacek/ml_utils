import numpy as np

class DataReader(object):
    def __init__(self,paresers=None, data_sep='#'):
        self.data_sep=data_sep
        self.paresers=paresers
        
    def __call__(self,out_path):
        raw_file=read_file(out_path)
        basic_data=basic_parse(raw_file,data_sep='#')
        n_samples=len(basic_data)
        def parser_helper(i):
            return [ self.parsers(basic_data[j][i])
                     for j in range(n_samples)]
        outputs=[ parser_helper(i)
               for i in range(len(self.paresers))]
        outputs[0]=np.array(outputs[0])
        return tuple(outputs)
#class DataReader(object):
#    def __init__(self, data_sep='#'):
#        self.data_sep=data_sep 
#        self.data_parser = Parser(parse_vector,0)
#        self.cat_parser = Parser(CatParser(),1)
#        self.person_parser = Parser(person_parser,2)

#    def __call__(self,out_path):
#        raw_file=read_file(out_path)
#        basic_data=basic_parse(raw_file,data_sep='#')
#        x=self.data_parser(basic_data)
#        y=self.cat_parser(basic_data)
#        persons=self.person_parser(basic_data)
#        X=np.array(x)
#        return X,y,persons

class Parser(object):
    def __init__(self,parse,row=0):
        self.row=row
        self.parse=parse
    
    def __call__(self,basic_data):
        if(len(basic_data[0])<(self.row+1)):
            return None
        return [ self.parse(inst_i[self.row])
                  for inst_i in basic_data]

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