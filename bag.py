import random,os.path
import unify,filtr,dataset,files

def bag_ens(in_path,size=6):
    cur_dir=os.path.split(in_path)[0]
    raw_ts=unify.read(in_path)
    out_path=cur_dir+'/bag'
    files.make_dir(out_path)
    for k in range(size):
        bag_k=sampled_dataset(raw_ts)
        out_k=out_path+'/sample'+str(k)
        bag_k.save(out_k)
    raw_ts.save(out_path+'/full')

def sampled_dataset(ts_dataset):
    train,test=filtr.split(ts_dataset.ts_names())
    sampled_names=random.sample(train,len(train))
    pairs=[ (name_i+'_'+str(i),ts_dataset[name_i]) 
                for i,name_i in enumerate(sampled_names)]
    pairs+=[ (name_i,ts_dataset[name_i]) for name_i in test]
    return dataset.TSDataset(dict(pairs) ,ts_dataset.name+'_bag')

bag_ens("../MSR/agum")