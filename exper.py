import feats
#import dataset
#import eval
#import select_feat
#from sklearn import preprocessing
#from sets import Set

def exper_single(in_path):
    feat_dataset=feats.read(in_path)
    train,test=split_data(feat_dataset)
    print(train.X.shape)
    print(test.X.shape)

def split_data(feat_dataset):
    train,test={},{}
    feat_dict=feat_dataset.to_dict()
    for name_i,data_i in feat_dict.items():
        if(person_selector(name_i)):
            train[name_i]=data_i
            print(name_i)
        else:
            test[name_i]=data_i
    return feats.from_dict(train),feats.from_dict(test)

def person_selector(name_i):
    return (int(name_i.split('_')[1])%2)==1 
#def experiment_restricted(paths,cats=[],cls_type='svm',to_zero=True):
#    if(to_zero):
#        cats=[ (cat_i-1) 
#               for cat_i in cats]
#    dataset=single_dataset(paths)   
#    r_data=dataset.select_category(data,cats)
#    odd_data,even_data=split_data(r_data)
#    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

#def experiment_basic(data,cls_type='svm'):
#    data=lasso_selection(paths[0],'pca',False)
#    print(data.y)
#    even_data,odd_data=split_data(data)
#    eval.determistic_eval(odd_data,even_data,cls_type=cls_type)

#def feat_selection(data,select=True,norm=True):
#    if(norm):
#        data=data(preprocessing.scale)        
#    if(select==False or select is None):
#        return data
#    n_feats=select[1]
#    if(data.dim()<n_feats):
#        print(select)
#        print("No selection only %d features required" % data.dim())
#        return data 
#    return select_feat.select_feat(data,select)

if __name__ == "__main__":
    exper_single("btf.txt")