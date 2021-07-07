import numpy as np
import feats,exp,ens

def make_dataset(paths,out_path,clf="LR"):
    models,datasets=ens.get_models(paths,clf=clf)
    prob_votes=[] 
    for model_i,data_i in zip(models,datasets):
        X_i=data_i.get_X()
        pred_i=model_i.predict_proba(X_i)
        prob_votes.append(pred_i)
    prob_votes=np.concatenate(prob_votes, axis=1)
    teacher_feats=feats.Feats()
    names=list(datasets[0].keys())
    for i,name_i in enumerate(names):
        teacher_feats[name_i]=prob_votes[i]
    teacher_feats.save(out_path)

def teacher_exp(in_path,n_cats=12):
    full_dataset=feats.read(in_path)[0]
    datasets=split_dataset(full_dataset,n_cats)
    votes=ens.make_votes(datasets,clf="LR")
    result=votes.voting(False)
    result.report()

def split_dataset(full_dataset,n_cats):
    split_size=int(full_dataset.dim()[0]/n_cats)
    datasets=[feats.Feats() for i in range(n_cats)]
    for name_i,x_i in full_dataset.items():
        for j,data_j in enumerate(datasets):
            data_j[name_i]=x_i[j*split_size:(j+1)*split_size]
    return datasets

dataset="ICCCI"
dir_path="../../2021_VI"
paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
print(paths)
#make_dataset(paths,"3DHOI")
student_path="../conv_frames/student/feats"
teacher_exp("3DHOI")