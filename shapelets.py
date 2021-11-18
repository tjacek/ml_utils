from tslearn.shapelets import LearningShapelets
import numpy as np
import files,seqs,feats
import timeit

def make_feats(in_path):
    model=train_model(in_path)
    def helper(name_i,data_i):
        print(name_i)
        print(data_i.shape)
        frames=seqs.inter(data_i,36)
        frames= np.squeeze(frames)
        frames=np.expand_dims(frames,axis=0)
        distances = model.transform(frames)
        return distances
    seqs.transform_lazy(in_path,helper,out_path="feats.txt")

def train_model(in_path):
    paths=select_paths(in_path)
#    paths=paths[:3] + paths[-3:]
    ts=seqs.from_paths(paths)
    ts=ts.split()[0]
    ts.resize(36)
#    raise Exception(ts.shape())
    model = LearningShapelets(n_shapelets_per_size=None,max_iter=2) #{3: 40})
    X,y,names=ts.as_dataset()
    model.fit(X,y)
    return model

def select_paths(in_path):
    return files.top_files(in_path)
#    paths=[]
#    persons=set(["4","5","6"])
#    for path_i in files.top_files(in_path):
#        name_i=files.get_name(path_i)
#        print(name_i)
#        if(name_i.get_person() %):
#            person_i=name_i.split("_")[2]
#            if(person_i in persons):
#                paths.append(path_i)
#    return paths

def compute_shaplets(in_path,out_path,n_feats=40):
    ts=seqs.read_seqs(in_path)
    ts.resize(64)
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

in_path="../common/shape_32"#"../conv_frames/seqs"
out_path="../common/feats"
#feat_exp(in_path,out_path)
start = timeit.timeit()
make_feats(in_path)
end = timeit.timeit()
print(end - start)