import numpy as np
import exper.cats,exper.selection,exper.inspect,learn
from sklearn.metrics import classification_report,accuracy_score
import files

def exp(args,in_path,dir_path=None,clf="LR",train=True):
    if(dir_path):
    	in_path+="/"+dir_path
    if(train):
        exper.cats.make_votes(args,in_path,clf_type=clf)
    exper.cats.adaptive_votes(in_path)

def show_acc_curve(in_path,dir_path=None,n_select=None):
    if(dir_path):
        in_path+="/"+dir_path
    ord=exper.selection.clf_selection(in_path)
    return exper.cats.acc_curve(in_path,ord)

def show_acc(in_path,dir_path=None):
    if(dir_path):
        in_path+="/"+dir_path
    print(exper.inspect.clf_acc(in_path))

def selection_result(vote_path,n_select,out_path=None):
    ord=exper.selection.clf_selection(vote_path)
    votes=exper.cats.selected_votes(vote_path,ord,binary=False)
    s_votes=votes[:n_select]
    result=exper.cats.simple_voting(s_votes)
    print(classification_report(result[1], result[0],digits=4))
    learn.show_confusion(result,out_path)

def to_csv(in_path,out_path):
    csv='name,accuracy,precision,recall,f1\n'
    for path_i in files.top_files(in_path):
        result_i=exper.cats.adaptive_votes(path_i,show=False) 
        csv+= "%s,%s,\n"% (path_i.split("/")[-1],result_i)
    file_str = open(out_path,'w')
    file_str.write(csv)
    file_str.close()

def selection_to_csv(in_path,out_path):
    csv='name,accuracy,precision,recall,f1\n'
    for path_i in files.top_files(in_path):
        ord=exper.selection.clf_selection(path_i)
        acc_i=exper.cats.acc_curve(path_i,ord,show=False)
        k=np.argmax(acc_i)
        result_i=get_result(path_i,ord,k+1)
        result_i=learn.compute_score(result_i[1],result_i[0])
        csv+= "%s,%d,%s,\n"% (path_i.split("/")[-1],k,result_i)
    file_str = open(out_path,'w')
    file_str.write(csv)
    file_str.close()

def get_result(vote_path,ord,n_select):
    votes=exper.cats.selected_votes(vote_path,ord,binary=False)
    s_votes=votes[:n_select]
    return exper.cats.simple_voting(s_votes)