import numpy as np
import feats,exp,ens,learn

def make_dataset(paths,out_path,clf="LR",fun=None):
    models,datasets=ens.get_models(paths,clf=clf)
    names=list(datasets[0].keys())        
    if(fun is None):
        fun=base
    prob_votes=[] 
    for model_i,data_i in zip(models,datasets):
        pred_i=fun(model_i,data_i,names)
        prob_votes.append(pred_i)
    prob_votes=np.concatenate(prob_votes, axis=1)
    teacher_feats=feats.Feats()
    for i,name_i in enumerate(names):
        teacher_feats[name_i]=prob_votes[i]
    teacher_feats.save(out_path)

def base(model_i,data_i,names):
    X_i=data_i.get_X(names)
    return model_i.predict_proba(X_i)
    
def hard_votes(model_i,data_i,names):
    pred_i=base(model_i,data_i,names)
    hard_pred=[]
    for x_j in pred_i:
        k=np.argmax(x_j)
        hard_j=np.zeros(x_j.shape)
        hard_j[k]=1
        hard_pred.append(hard_j)
    return np.array(hard_pred)

def teacher_exp(in_path,n_cats=12):
    full_dataset=feats.read(in_path)[0]
    datasets=split_dataset(full_dataset,n_cats)
    result=voting(datasets)
    result.report()

def split_dataset(full_dataset,n_cats):
    split_size=int(full_dataset.dim()[0]/n_cats)
    datasets=[feats.Feats() for i in range(n_cats)]
    for name_i,x_i in full_dataset.items():
        for j,data_j in enumerate(datasets):
            data_j[name_i]=x_i[j*split_size:(j+1)*split_size]
    return datasets

def voting(datasets):
    datasets= [data_i.split()[1] for data_i in datasets]
    y_true,y_pred=[],[]
    names=list(datasets[0].keys())
    for name_i in names:
        votes_i=[data_j[name_i] for data_j in datasets ]
        votes_i=np.array(votes_i)
        cat_i=  np.argmax(np.sum( votes_i,axis=0))
        y_pred.append(cat_i)
        y_true.append(name_i.get_cat())
    return learn.Result(y_true,y_pred,names)

dataset="ICCCI"
dir_path="../../2021_VI"
paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
print(paths)
make_dataset(paths,"3DHOI_hard",fun=hard_votes)
#student_path="../conv_frames/student_hard/feats"
#teacher_exp(student_path)