import dataset
import eval
import select_feat
from sklearn import preprocessing

def exper_single(paths,norm=False,select=False,cls_type='svm'):
    data=lasso_selection(paths,select,norm)
    even_data,odd_data=split_data(data)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def experiment_restricted(paths,cats=[],cls_type='svm',to_zero=True):
    if(to_zero):
        cats=[ (cat_i-1) 
               for cat_i in cats]
    data=lasso_selection(paths,select='lasso')
    r_data=dataset.select_category(data,cats)
    odd_data,even_data=split_data(r_data)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def experiment_basic(paths,select,norm,cls_type='svm'):
    data=dataset.read_and_unify(paths,select,norm)
    even_data,odd_data=split_data(data)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def lasso_selection(paths,select=True,norm=True):
    data=single_dataset(paths)
    if(norm):
        data=data(preprocessing.scale)        
    if(select!=False):
        data=select_feat.select_feat(data,select)
    return data

def split_data(r_data):
    even_data=dataset.select_person(r_data,i=0)
    odd_data=dataset.select_person(r_data,i=1)
    return even_data,odd_data

def single_dataset(paths):
    all_datasets=[ dataset.get_dataset(path_i) 
                   for path_i in paths]
    data=dataset.unify_feat(all_datasets)
    return data

if __name__ == "__main__":
    #in_path1= "../exper2/united/dataset.txt"
    #in_path2= "../exper2/proj/dataset.txt"

    in_path1= '../exper_4/dataset.txt'
    in_path2= '../final_paper/MSRaction/simple/dataset.txt'
    in_path3='../final_paper/MSRaction/basic_nn/dataset.txt'
    paths=[in_path1,in_path2,in_path3]#,in_path3]
    #exper_single(paths,norm=True,select='lasso',cls_type='svm')
    A1=[2,3,5,6,10,13,18,20]
    A2=[1,4,7,8,9,11,12,14]
    A3=[6,14,15,16,17,18,19,20]
    A=[i for i in range(20)]
    experiment_restricted(paths,cats=A3,cls_type='svm')
    