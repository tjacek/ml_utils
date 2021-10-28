from tslearn.shapelets import LearningShapelets
import files,seqs,feats

def compute_shaplets(in_path,out_path,n_feats=40):
    ts=seqs.read_seqs(in_path)
    ts.subsample(36)
#    raise Exception(ts.shape())
#    ts.resize(64)
    train,test=ts.split()
    model = LearningShapelets(n_shapelets_per_size={3: n_feats})
    train_X,train_y,train_names=train.as_dataset()
    model.fit(train_X,train_y)
    X,y,names=ts.as_dataset()
    print(X.shape)
    distances = model.transform(X)
    dist_feat=feats.Feats()
    for i,x_i in enumerate(distances):
        dist_feat[names[i]]=x_i
        print(x_i.shape)	
    dist_feat.save(out_path)

def feat_exp(in_path,out_path,n=20,step=10):
    files.make_dir(out_path)
    for i in range(1,n+1):
        n_feats=i*step
        out_i="%s/%d" % (out_path,n_feats)	
        compute_shaplets(in_path,out_i,n_feats= n_feats)	


in_path="../conv_frames/seqs"
out_path="../conv_frames/feats"
feat_exp(in_path,out_path)
