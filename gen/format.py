import sys
sys.path.append("..")
import exp,files

class LineDict(dict):
    def __init__(self, arg=[]):
        super(LineDict, self).__init__(arg)

    def get_ids(self):
    	return set([",".join(key_i.split(",")[:-1]) 
	              for key_i in self.keys()])
    
    def check_improv(self,id_i):
        base,opv=self.get_pair(id_i)
        return 0<(opv.accuracy-base.accuracy)
    
    def get_pair(self,id_i):
        return self[f"{id_i},base"],self[f"{id_i},opv"]	

def make_table(paths):
    all_exps=[read_exp(path_i) 
	        for path_i in paths]
    all_ids=list(all_exps[0].get_ids())
    
    improv=[id_i for id_i in all_ids
               if(all_exps[0].check_improv(id_i))]
    print(improv)
#	print(all_exps[0].get_pair(all_ids[0]))

def read_exp(in_path):
    name=in_path.split("/")[-2]
    lines=exp.read_lines(in_path,as_class=True)
    for line_i in lines:
    	line_i.info.append(name)
    return LineDict({line_i.desc:line_i for line_i in lines})

paths=["binary/full.csv","no_grid/full.csv","grid/full.csv",]
lines=make_table(paths)