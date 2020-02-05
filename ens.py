import exper.cats,exper.selection,exper.inspect,learn
from sklearn.metrics import classification_report,accuracy_score

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

def selection_result(vote_path,n_select):
    ord=exper.selection.clf_selection(vote_path)
    votes=exper.cats.selected_votes(vote_path,ord,binary=False)
    s_votes=votes[:n_select]
    result=exper.cats.simple_voting(votes)
    print(classification_report(result[1], result[0],digits=4))