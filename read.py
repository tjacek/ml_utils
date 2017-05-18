import numpy as np

class DataReader(object):
    def __init__(self, data_sep='#'):
        self.data_sep=data_sep 
        self.data_parser = Parser(parse_vector,0)
        self.cat_parser = Parser(cat_parser,1)
        self.person_parser = Parser(cat_parser,2)

    def __call__(self,out_path):
        raw_file=read_file(out_path)
        basic_data=basic_parse(raw_file,data_sep='#')
        x=self.data_parser(basic_data)
        y=self.cat_parser(basic_data)
        persons=self.person_parser(basic_data)
        X=np.array(x)
        return X,y,persons

class Parser(object):
    def __init__(self,parse,row=0):
        self.row=row
        self.parse=parse
    
    def __call__(self,basic_data):
        if(len(inst_i[0])<(self.row+1)):
            return None
        return [ self.parse(inst_i[self.row])
                  for inst_i in basic_data]

def parse_vector(raw_data):       
    return [ float(x_i) 
             for x_i in raw_data.split(',')] 

def cat_parser(raw_cat):       
    return int(raw_cat)


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