import dataset
import eval
import select_feat

def experiment_full(in_path,in_path2,select=True,cls_type='svm'):
    data=dataset.read_and_unify(in_path,in_path2,select=select)
    even_data=dataset.select_person(data,i=0)
    odd_data=dataset.select_person(data,i=1)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def experiment_single(in_path,select=True,cls_type='svm'):
    data=dataset.get_annotated_dataset(in_path)
    if(select):
        data=select_feat.lasso_model(data)
    even_data=dataset.select_person(data,i=0)
    odd_data=dataset.select_person(data,i=1)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def experiment_restricted(in_path,in_path2,cats=[],from_zero=False):
    data=dataset.read_and_unify(in_path,in_path2)
    if(from_zero):
        cats=[(cat_i-1) for cat_i in cats]
    r_data=dataset.select_category(data,cats)
    even_data=dataset.select_person(r_data,i=0)
    odd_data=dataset.select_person(r_data,i=1)
    eval.determistic_eval(odd_data,even_data,svm=False)

def experiment_basic(paths,select,norm,cls_type='svm'):
    data=dataset.read_and_unify(paths,select,norm)
    even_data,odd_data=split_data(data)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def split_data(r_data):
    even_data=dataset.select_person(r_data,i=0)
    odd_data=dataset.select_person(r_data,i=1)
    return even_data,odd_data

if __name__ == "__main__":
    #in_path1= "../exper2/united/dataset.txt"
    #in_path2= "../exper2/proj/dataset.txt"
    in_path1='../final_paper/MSRaction/basic_nn/dataset.txt'
    #in_path1= '../exper2/time/dataset.txt'
    in_path2= '../final_paper/MSRaction/simple/dataset.txt'
    #"../../final_paper/MSRaction/simple_dataset.txt"
    #"../exper_s/old/dataset_recr.txt"
    #experiment_single(in_path,select=True,cls_type='rf')
    paths=[in_path1]#,in_path2]
    select=[True]#,False]#,False]
    norm=[True]#,False]#,False]
    experiment_basic(paths,select,norm,cls_type='rf')
    