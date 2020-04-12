import matplotlib.pyplot as plt
import exper.selection
import learn,feats,exper.cats,files

def acc_curve(vote_path,clf_ord="best",binary=False,show=True):
    acc=exper.selection.get_acc(vote_path,clf_ord)
    title=vote_path.split('/')[-1]
    show_curve(acc,len(acc),title)
    return acc

def acc_to_csv(in_path,out_path,fun):
    names,acc=["Accuracy for"],[]
    for path_i in files.top_files(in_path):
        name_i=path_i.split('/')[-1]
        names.append(name_i)
        acc.append(fun(path_i))
    acc=list(zip(*acc))
    csv=",".join(names)+'\n'
    for i,acc_i in enumerate(acc):
        acc_i=[round(acc_ij,4) for acc_ij in acc_i]
        csv+= "%d,%s\n" % (i+1,",".join(map(str,acc_i)))
    file_str = open(out_path,'w')
    file_str.write(csv)
    file_str.close()

def all_curves(in_path,out_path,fun):
    files.make_dir(out_path)
    for path_i in files.top_files(in_path):
        name_i=path_i.split('/')[-1]
        acc_i=fun(path_i)
        out_i="%s/%s" % (out_path,name_i)
        n_clf=len(acc_i)
        show_curve(acc_i,n_clf,name_i,out_i)

def show_curve(acc,n_clf,name,out_path=None):
    plt.title(name)
    plt.grid(True)
    plt.xlabel('number of classifiers')
    plt.ylabel('accuracy')
    plt.plot(range(1,n_clf+1), acc, color='red')
    if(out_path):
        plt.savefig(out_path)
    else:    
        plt.show()
    plt.clf()
    return acc