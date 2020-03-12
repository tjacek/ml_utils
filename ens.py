import numpy as np,os.path,os
import exper.cats,exper.selection,exper.inspect,exper.curve
from sklearn.metrics import classification_report,accuracy_score
from boost.ada_boost import ada_boost
import files,feats,learn

def exp(args,in_path,dir_path=None,clf="LR",train=True):
    if(dir_path):
    	in_path+="/"+dir_path
    if(train):
        exper.cats.make_votes(args,in_path,clf_type=clf,train_data=True)
    exper.cats.adaptive_votes(in_path,binary=False)

#def show_acc_curve(in_path,dir_path=None,n_select=None):
#    if(dir_path):
#        in_path+="/"+dir_path
#    ord=exper.selection.clf_selection(in_path)
#    return exper.curve.acc_curve(in_path,ord)

def show_acc(in_path,dir_path=None):
    if(dir_path):
        in_path+="/"+dir_path
    print(exper.inspect.clf_acc(in_path))

#def selection_result(vote_path,n_select,out_path=None):
#    ord=exper.selection.clf_selection(vote_path)
#    votes=exper.cats.selected_votes(vote_path,ord,binary=False)
#    s_votes=votes[:n_select]
#    result=exper.cats.simple_voting(s_votes)
#    print(classification_report(result[1], result[0],digits=4))
#    learn.show_confusion(result,out_path)

def to_csv(in_path,out_path):
    def helper(path_i):
        stats=exper.cats.adaptive_votes(path_i,show=False) 
        return "ALL,"+stats
    return to_csv_template(in_path,out_path,helper)

def selection_to_csv(in_path,out_path):
    def helper(path_i):
        clf_ord=exper.selection.clf_selection(path_i)
        print(ord)
        results=exper.selection.selected_voting(path_i,clf_ord)
        acc_i=learn.acc_arr(results)
        k=np.argmax(acc_i)
        result_i=results[k]
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

#def res_corl_matrix(in_path,out_path):
#    if(os.path.isdir(in_path)):
#        files.make_dir(out_path) 
#        for i,path_i in enumerate(files.top_files(in_path)):
#            print(path_i)
#            out_i=out_path+'/'+path_i.split("/")[-1]
#            X=exper.selection.get_res_corelation(path_i)
#            np.savetxt(out_i, X, fmt='%.2e', delimiter=',')

#import exper.persons

#def unify_common(feat_path,out_path,n_feats):
#    common=feats.read(feat_path)
#    common.norm()
#    if(n_feats>0):
#        common=common.reduce(n_feats)
#    train_i,test_i=common.split()
#    test_votes=exper.persons.pred_vectors(train_i,test_i,"LR")  
#    train_votes=exper.persons.pred_by_person(train_i,"LR")
#    votes_dict=dict(train_votes+test_votes)   
#    votes_feats=feats.from_dict(votes_dict)
#    votes_feats.save(out_path)
