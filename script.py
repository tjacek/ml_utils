import ens,files,exper.cats

clf_type="LR"
feats=["stats","basic","sim"]
seq_type="ens2"
train=False

files.make_dir(seq_type)
files.make_dir("%s/%s" % (seq_type,clf_type))
for feat_i in feats:
    for feat_j in feats:
        feat_path="../%s/%s/feats" % (seq_type,feat_j)
        hc_path="../proj/%s/feats" % feat_i
        vote_path="%s/%s/%s_%s"% (seq_type,clf_type,feat_i,feat_j)        
        args={'hc_path':hc_path,'deep_paths':feat_path,'n_feats':0}        
        print(vote_path)
        if(train):
            ens.exp(args,vote_path,clf=clf_type,train=train,show=True)
        else:
            result=exper.cats.adaptive_votes(vote_path,binary=False,show=False)
            print(result)
