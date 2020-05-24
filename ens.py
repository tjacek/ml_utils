import numpy as np,os.path,os
import exper.cats,exper.selection,exper.inspect,exper.curve
from sklearn.metrics import classification_report,accuracy_score
from boost.ada_boost import ada_boost
import files,feats,learn

def exp(args,in_path,dir_path=None,clf="LR",train=True):
    if(dir_path):
    	in_path+="/"+dir_path
    if(train):
        exper.cats.make_votes(args,in_path,clf_type=clf,train_data=False)
    exper.cats.adaptive_votes(in_path,binary=False)

def show_acc_curve(in_path,clf_ord=None):
    if(not clf_ord):
        clf_ord=exper.selection.clf_selection(in_path)
    return exper.curve.acc_curve(in_path,clf_ord)

def show_acc(in_path,clf_ord=None):
    print(exper.inspect.clf_acc(in_path))

def to_csv(in_path,out_path):
    def helper(path_i):
        stats=exper.cats.adaptive_votes(path_i,show=False) 
        return "ALL,"+stats
    return to_csv_template(in_path,out_path,helper)

def selection_to_csv(in_path,out_path):
    def helper(path_i):
        result_i=selection_result(path_i)
        result_i=learn.compute_score(result_i[1],result_i[0])
        return "%d,%s" % (k,result_i)
    return to_csv_template(in_path,out_path,helper)

def ada_to_csv(in_path,out_path):
    def helper(path_i):
        y_true,y_pred,names=ada_boost(path_i,show=False)
        stats=learn.compute_score(y_true,y_pred,as_str=True)
        print(stats)
        return "ALL,"+stats
    return to_csv_template(in_path,out_path,helper)

def to_csv_template(in_path,out_path,fun):
    csv='name,n_clfs,accuracy,precision,recall,f1\n'
    for path_i in files.top_files(in_path):
        result_i=fun(path_i)
        csv+= "%s,%s,\n"% (path_i.split("/")[-1],result_i)
    file_str = open(out_path,'w')
    file_str.write(csv)
    file_str.close()

def cf_matrix(in_path,out_path,selection=True):
    files.make_dir(out_path) 
    for i,path_i in enumerate(files.top_files(in_path)):
        print(path_i)
        if(selection):
            result_i=exper.selection.selection_result(path_i)
        else:
            clf_ord=exper.selection.clf_selection(path_i)
            result_i=exper.selection.selected_voting(path_i,clf_ord)[-1]
        out_i="%s/%s"% (out_path,path_i.split("/")[-1])
        learn.show_confusion(result_i,out_i)
