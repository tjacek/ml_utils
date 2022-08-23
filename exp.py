import numpy as np
import files,ens
from dataclasses import dataclass,field
from functools import wraps
import pandas as pd

@dataclass
class Line:
    desc:str
    accuracy:float
    precision:float   
    recall:float 
    f1_score:float
    info:list=field(default_factory=list)

    def __str__(self):
        if(len(self.info)==1):
            pref=self.info[0]
        else:
            pref=",".join(self.info)
        metrics=f"{self.accuracy},{self.precision},{self.recall},{self.f1_score}"
        return f"{self.desc},{pref},{metrics}" 

def save_lines(lines,out_path):
    print(lines)
    if(out_path):
        txt="\n".join(lines)
        out_file = open(out_path,"w")
        out_file.write(txt)
        out_file.close()

def read_lines(in_path,as_class=True):
    lines=[]
    with open(in_path,"r") as in_file:  
        for line_i in in_file.readlines():
            if(as_class):
                line_i=line_i.split(",")
                acc,prec,recall,f1_score=[float(m) for m in line_i[-4:]]
                desc=",".join(line_i[:-4])
                line_i=Line(desc,acc,prec,recall,f1_score)
            lines.append(line_i)
    return lines

def get_metrics(result_i):
    acc_i= result_i.get_acc()
    metrics="%.4f,%.4f,%.4f" % result_i.metrics()[:3]
    return "%.4f,%s" % (acc_i,metrics)

def order_lines(in_path,out_path=None,col='Split'):
    df=pd.read_csv(in_path)
    df.sort_values(by=col,inplace=True)
    df = df.reset_index(drop=True)
    if(out_path):
        df.to_csv(out_path)
    else:
        print(df)
#    for line_i in lines:

#class MultiEnsembleExp(object):
#    def __init__(self,all_ensembles,threshold=0.05):
#        self.all_ensembles=all_ensembles
#        self.threshold=threshold

#    def __call__(self,input_dict,out_path=None):
#        lines=[]
#        for desc_i,ensemble_i in self.all_ensembles.items():
#            result_i,votes_i=ensemble_i(input_dict)
#            if(type(votes_i)==ens.Votes):
#                n_clf=len(votes_i)
#            elif(type(votes_i)==int):
#                n_clf=votes_i
#            else:
#                n_clf=votes_i[votes_i>self.threshold].shape[0] 
#            line_i="%s,%d,%s" % (desc_i,n_clf,get_metrics(result_i))
#            lines.append(line_i)
#        save_lines(lines,out_path)
#        return lines

#class EnsembleExp(object):
#    def __init__(self,ensemble=None,gen=None):
#        if(ensemble is None):
#            ensemble=ens.Ensemble()
#        if(gen is None):
#            gen=simple_gen
#        self.ensemble=ensemble
#        self.gen=gen
        
#    def __call__(self,input_dict,out_path=None):
#        lines=[]
#        for desc_i,path_i in self.gen(input_dict):
#            print(path_i)
#            result_i=self.ensemble(path_i)[0]
#            line_i="%s,%s" % (desc_i,get_metrics(result_i))
#            lines.append(line_i)
#        save_lines(lines,out_path)
#        return lines

def save_results(fun):
    @wraps(fun)
    def helper(out_path,*args, **kwargs):
        result_dict=fun(*args, **kwargs)
        if(type(result_dict)!=dict):
            result_dict=dict(result_dict)
        lines=[]
        for name_i,result_i in result_dict.items():
            line_i=f'{name_i},{get_metrics(result_i)}'
            lines.append(line_i)
        save_lines(lines,out_path)
    return helper

@files.dir_function(args=1)
def acc_exp(in_path):
    print(in_path)
    results,ensemble=[],ens.EnsembleHelper()
    for path_i in files.top_files(in_path):
        paths=(f"{path_i}/common",f"{path_i}/binary")
        results.append(ensemble(paths)[0])
    acc=[result_i.get_acc() for result_i in results]
    acc_mean,acc_std=np.mean(acc),np.std(acc)
    return in_path.split("/")[-1],(acc_mean,acc_std)

if __name__ == "__main__":
#    lines= read_lines("inert.csv")
#    print(lines)
    order_lines("inert.csv","inert2.csv",col=['Split','Accuracy'])