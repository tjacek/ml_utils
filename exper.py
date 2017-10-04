import dataset
import eval
import select_feat
from sklearn import preprocessing
from sets import Set

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
    dataset=single_dataset(paths)   
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
    n_feats=select[1]
    if(data.dim()<n_feats):
        print(select)
        print("No selection only %d features required" % data.dim())
        return data
    if(select!=False):
        data=select_feat.select_feat(data,select)
    return data

def split_data(r_data):
    even_data=dataset.select_person(r_data,i=0)
    odd_data=dataset.select_person(r_data,i=1)
    return even_data,odd_data

def single_dataset(paths,select=None):
                        #[1,2,4,5,9,12,17,19]
                        #[0,3,6,7,8,10,11,13] 
                        #[5,13,14,15,16,17,18,19]
    if(type(paths)==str):
        paths=read_paths(paths)
    all_datasets=[ dataset.get_dataset(path_i) 
                   for path_i in paths]
    if(select!=None):
        all_datasets=[dataset.select_category(data_i,select) 
                          for data_i in all_datasets]
    for data_i in all_datasets:
        print("Orginal size" + str(data_i.X.shape))
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

    #paths=read_paths('conf/no_.txt')
    #out_path="../AArtyk/binary_time/cat4/simple.txt"
    paths=['../AArtyk/simple/all/simple.txt','../AArtyk/simple/max_z/dtw_dataset.txt']
    exper_single(paths,norm=True,select=('rfe',150),cls_type='svm')