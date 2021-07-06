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

dataset="ICCCI"
dir_path="../../2021_VI"
paths=exp.basic_paths(dataset,dir_path,"dtw","ens_splitI/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
print(paths)
make_dataset(paths,"3DHOI")