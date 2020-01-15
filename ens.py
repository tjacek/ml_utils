import exper.cats,exper.selection,exper.inspect

def exp(args,in_path,dir_path=None,clf="LR",train=True):
    if(dir_path):
    	in_path+="/"+dir_path
    if(train):
        exper.cats.make_votes(args,in_path,clf_type=clf)
    exper.cats.adaptive_votes(in_path)

def show_acc_curve(in_path,dir_path=None):
    if(dir_path):
        in_path+="/"+dir_path
    ord=exper.selection.clf_selection(in_path)
    acc=exper.cats.acc_curve(in_path,ord)

def show_acc(in_path,dir_path=None):
    if(dir_path):
        in_path+="/"+dir_path
    print(exper.inspect.clf_acc(in_path))