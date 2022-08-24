import numpy as np
import files,ens
from dataclasses import dataclass,field
from functools import wraps
import re
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

def to_csv(fun):
    @wraps(fun)
    def helper(in_path,out_path=None,info=None):
        df=pd.read_csv(in_path)
        df=fun(df,info)
        if(out_path):
            if(type(df)==list):
                save_lines(df,out_path)
            else:
                df.to_csv(out_path)
        else:
            print(df)
    return helper

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

@to_csv
def order_lines(df,col='Split'):
    df.sort_values(by=col,inplace=True)
    df = df.reset_index(drop=True)
    return df

@to_csv
def transform_cols(df,info):
    get_digit=lambda text:re.sub(r'[^\d+]','',text)
    df['Features']=df['Features'].map(get_digit)
    return df

@to_csv
def square(df,info):
    splits= df['Split'].unique()
    feats= df['Features'].unique()
    lines=[ ','.join(['dataset']+splits)]
    for feat_i in feats:
        line_i=[feat_i]
        df_i=df[df['Features']==feat_i]
        for split_j in splits:
            acc=df_i[ df_i['Split']==split_j]['Accuracy']
            line_i.append(str(float(acc)))
        lines.append(','.join(line_i))
    print(lines)
    return lines

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
#    order_lines("inert.csv","inert2.csv",col=['Split','Features'])
    square("inert2.csv",out_path='square.csv')#"inert3.csv")