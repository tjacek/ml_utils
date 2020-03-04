import numpy as np
import files,feats,exper,learn

def get_args_dict(path):
    return {'hc_path':None,'deep_paths':path,'n_feats':0}

def get_data(args):
    if(type(args)==str):
        args={'hc_path':None,'deep_paths':args,'n_feats':0}
    return get_datasets(**args,norm=True )

def get_datasets(hc_path,deep_paths,n_feats,norm=True ):
    if(not n_feats):
        n_feats=0
    (n_hc_feats,n_deep_feats)= (n_feats,None) if(type(n_feats)==int) else n_feats
    hc_feats=read_hc(hc_path,n_hc_feats)
    if(not deep_paths):
        return [hc_feats]    
    full_feats=[]
    for path_i in files.top_files(deep_paths):
        print(path_i)
        deep_i=feats.read(path_i)
        if(norm):
            deep_i.norm()
        if(n_deep_feats):
            deep_i.reduce(n_deep_feats)
        full_i= (hc_feats + deep_i) if(hc_feats) else deep_i
        full_feats.append( full_i)
    return full_feats

def read_hc(hc_path,n_feats):
    if(not hc_path):
        return None
    hc_feats=feats.read(hc_path)
    hc_feats.norm()
    if(n_feats):
        hc_feats.reduce(n_feats)
    return hc_feats

def select_feats(in_path,out_path,n_feats=130):
    datasets=get_datasets(None,in_path,(0,n_feats))
    files.make_dir(out_path)
    for i,data_i in enumerate(datasets):
        out_i=out_path+'/feats'+str(i)
        data_i.save(out_i)
