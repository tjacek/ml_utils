import numpy as np
import random,os.path
import unify,filtr,dataset,files

def bag_ens(in_path,size=6):
    out_path,raw_ts=prepare_date(in_path,name='/bag')
    for k in range(size):
        bag_k=sampled_dataset(raw_ts)
        out_k=out_path+'/sample'+str(k)
        bag_k.save(out_k)
    raw_ts.save(out_path+'/full')

def prepare_date(in_path,name='/bag'):
    cur_dir=os.path.split(in_path)[0]
    raw_ts=unify.read(in_path)
    out_path=cur_dir+name
    files.make_dir(out_path)
    return out_path,raw_ts

def sampled_dataset(ts_dataset):
    train,test=filtr.split(ts_dataset.ts_names())
    sampled_names=random.sample(train,len(train))
    pairs=[ (name_i+'_'+str(i),ts_dataset[name_i]) 
                for i,name_i in enumerate(sampled_names)]
    pairs+=[ (name_i,ts_dataset[name_i]) for name_i in test]
    return dataset.TSDataset(dict(pairs) ,ts_dataset.name+'_bag')

def jackknife(in_path):
    out_path,raw_ts=prepare_date(in_path,name='/jack')
    for k in range(raw_ts.n_feats()):
        sub_k=subspace_dataset(k,raw_ts)
        out_k=out_path+'/knief_'+str(k)
        sub_k.save(out_k,as_txt=False)
    raw_ts.save(out_path+'/full',as_txt=False)

def subspace_dataset(k,ts_dataset):
    names=ts_dataset.ts_names()
    sub_dict={}
    for name_i in names:
        feats_i=ts_dataset.as_features(name_i)
        s_feats_i=[ts_j 
                    for j,ts_j in enumerate(feats_i)
                        if(j!=k)]
        sub_dict[name_i]=np.array(s_feats_i).T
        print(ts_dataset[name_i].shape)
        raise Exception(sub_dict[name_i].shape)
    return dataset.TSDataset(sub_dict,ts_dataset.name+'_sub')

def person_ens(in_path,size=6):
    out_path,raw_ts=prepare_date(in_path,name='/person')
    names=raw_ts.ts_names() 
    train,test=filtr.split(names)   
    one_persons=filtr.by_person(train)
    files.make_dir(out_path)
    for i,person_i in enumerate(one_persons):
        names_i=person_i+test
        ts_i=raw_ts.select(names_i)
        out_i=out_path+'/person'+str(i)
        ts_i.save(out_i)

person_ens("../MSR/agum")