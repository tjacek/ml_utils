import exper.cats,exper.selection,exper.inspect

def show_acc_curve(in_path,dir_path=None):
    if(dir_path):
        in_path+="/"+dir_path
    ord=exper.selection.clf_selection(in_path)
    acc=exper.cats.acc_curve(in_path,ord)

def show_acc(in_path,dir_path=None):
    if(dir_path):
        in_path+="/"+dir_path
    print(exper.inspect.clf_acc(in_path))