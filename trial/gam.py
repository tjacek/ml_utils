import exper.voting
import files,feats
import numpy as np
from pygam import LogisticGAM,s, f
from pygam.terms import SplineTerm
from pygam.terms import TermList

def make_votes(args,out_path):
    datasets=exper.voting.get_data(args)
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
        train,test=data_i.split()
        cls_i=make_cls(train)
        pred_i=np.array([cls_ij.predict_proba(data_i
        	.X) 
        	                    for cls_ij in cls_i]).T
        print(pred_i.shape)
        out_i=out_path+'/nn'+str(i)
        feat_i=feats.FeatureSet(pred_i,data_i.info)
        feat_i.save(out_i)
    
def make_cls(train):
    y=np.array(train.get_labels())
    all_cls=[]
    for cat_i in range(train.n_cats()):
        y_i=np.zeros(y.shape)
        y_i[y==cat_i]=1.0
        n_feats=train.dim()
        terms=TermList(*[SplineTerm(feat,n_splines=6,penalties='l2' ) 
    	                for feat in range(n_feats)])
        cls_i=LogisticGAM(terms=terms,max_iter=20)
        print(train.X.shape)
        cls_i.fit(train.X,y_i)
        all_cls.append(cls_i)
    return all_cls

deep_path="../MHAD/ens2/feats"
args={'hc_path':"../MHAD/mean",'deep_paths':deep_path,'n_feats':(0,100)}

make_votes(args,'test2')
#from pygam.datasets import toy_classification
#X,y=toy_classification(return_X_y=True, n=50)

#print( type(s))