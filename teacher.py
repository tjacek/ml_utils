import numpy as np
import feats,exp,ens,learn,files

def make_dataset(paths,out_path,clf="LR",fun=None):
    if(type(paths)==str):
        paths={"binary":paths,"common":None}
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

def ground_votes(model_i,data_i,names):
    pred_i=base(model_i,data_i,names)
    ground_pred=[]
    for j,x_j in enumerate(pred_i):
        true_cat= names[j].get_cat()
        k=np.argmax(x_j)
        if(true_cat!=k):
            ground_j=np.zeros(x_j.shape)
            ground_j[true_cat]=1
        else:
            ground_j=x_j
        ground_pred.append(ground_j)
    return ground_pred

def label_smoothing(model_i,data_i,names,alpha=0.25):
    pred_i=base(model_i,data_i,names)
    smooth_pred=[]
    for x_j in pred_i:
        smooth_j=np.ones(x_j.shape)
        smooth_j/=x_j.shape[0]
        smooth_j= (1-alpha)*x_j + alpha*smooth_j
        smooth_pred.append(smooth_j)
    return smooth_pred

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

def dataset_exp(paths,out_path):
    files.make_dir(out_path)
    fun_dir={  "base":base,"hard":hard_votes,
        "ground":ground_votes,"smooth":label_smoothing}
    for name_i,fun_i in fun_dir.items():
        make_dataset(paths,"%s/%s" % (out_path,name_i),fun_i)

in_path="../conv_frames/test/simple_feats"
make_dataset(in_path,"3DHOI_simple")
#dataset="3DHOI"
#dir_path=".."
#paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
#paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
#print(paths)
#dataset_exp(paths,dataset)
#student_path="../conv_frames/student_ground/feats"
#teacher_exp(student_path)
