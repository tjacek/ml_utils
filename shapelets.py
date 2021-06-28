from tslearn.shapelets import LearningShapelets
import seqs,feats

def compute_shaplets(in_path,out_path):
    ts=seqs.read_seqs(in_path)
    ts.resize(64)
    train,test=ts.split()
    model = LearningShapelets(n_shapelets_per_size={3: 40})
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

in_path="../MSR/max_z/seqs"
out_path="../MSR/max_z/shape"
compute_shaplets(in_path,out_path)