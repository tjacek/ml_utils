import dataset
import eval
import select_feat
from sklearn import preprocessing

def exper_single(paths,norm=False,select=False,cls_type='svm'):
    data=single_dataset(paths)
    data=lasso_selection(data,select,norm)
    print(data.X.shape)
    even_data,odd_data=split_data(data)
    print(even_data.X.shape)
    print(odd_data.X.shape)
    print(even_data.y)
    print(odd_data.y )   
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def experiment_restricted(paths,cats=[],cls_type='svm',to_zero=True):
    if(to_zero):
        cats=[ (cat_i-1) 
               for cat_i in cats]
    data=lasso_selection(paths,select='lasso')
    r_data=dataset.select_category(data,cats)
    odd_data,even_data=split_data(r_data)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def experiment_basic(data,cls_type='svm'):
#    data=lasso_selection(paths[0],'pca',False)
    print(data.y)
    even_data,odd_data=split_data(data)
    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

def lasso_selection(data,select=True,norm=True):
    if(norm):
        data=data(preprocessing.scale)        
    if(data.dim()<select):
        return data
    if(select!=False):
        data=select_feat.select_feat(data,select)
    return data

def split_data(r_data):
    even_data=dataset.select_person(r_data,i=0)
    odd_data=dataset.select_person(r_data,i=1)
    return even_data,odd_data

def single_dataset(paths):
    if(type(paths)==str):
        paths=read_paths(paths)
    all_datasets=[ dataset.get_dataset(path_i) 
                   for path_i in paths]
    for data_i in all_datasets:
        print(data_i.X.shape)
        #data_i.y,data_i.person=data_i.person,data_i.y
    data=dataset.unify_feat(all_datasets)
    return data

def read_paths(conf_file):
    return [ line_i.replace('\n','') 
             for line_i in open(conf_file,"r")
                if(line_i!='')]

if __name__ == "__main__":
    #in_path_s='../konf/full/dataset.txt'
    #in_path_s2='../konf/simple/dataset.txt'

    #in_path1= '../exper_4/dataset.txt'
    #in_path2= '../final_paper/MSRaction/simple/dataset.txt'
    #in_path3='../final_paper/MSRaction/basic_nn/dataset.txt'
    #paths=[in_path1,in_path2,in_path_s]#
    
    #in_path0="../konf1/dtw_feat.txt"
    #in_path1="../konf1/dataset.txt"
    #in_path2="../konf1/simple.txt"

    #in_path1='../konf3/dtw_feats.txt'
    #in_path2='../konf3/simple.txt'
    #in_path3='../konf3/max_z.txt'

    #in_path1="../methods/Vb/dtw_feats.txt"
    #in_path2="../methods/Vb/hf_feats.txt"
    #in_path='../konf4/dtw_feats.txt'
    
    in_path3="../AArtyk/untime/nn/dtw_dataset.txt"
    #in_path3= '../final_paper/MSRaction/simple/dataset.txt'
    #in_path3= '../exper_4/dataset.txt'
    in_path1="../AArtyk/simple/max_z/dtw_dataset.txt"
    in_path2="../AArtyk/simple/corl/dtw_dataset.txt"
    #in_path3="../AArtyk/simple/skew/dtw_dataset.txt"
    
    in_path_s="../AArtyk/simple/all/simple.txt"

    #paths=read_paths('conf/exp.txt')
    out_path="../AArtyk/binary_time/cat4/simple.txt"
    exper_single([out_path],norm=True,select=150,cls_type='svm')