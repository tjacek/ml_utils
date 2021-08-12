import numpy as np
import learn,files,exp

def exp1(path1,path2):
    paths1=files.top_files(path1)
    paths2=files.top_files(path2)
    full_paths=[[p0_i,p1_i] for p0_i,p1_i in zip(paths1,paths2)]
    results=[learn.train_model(path_i) 
                for path_i in full_paths]
    acc=[ result_i.get_acc() for result_i in results]
    i=np.argmax(acc)
    print(acc)
    print(i)
    results[i].report()

def prepare_paths(dir_path,dtw="dtw",nn="1D_CNN",binary="ens_splitI"):
    common="%s/%s" % (dir_path,dtw)
    common=files.get_paths(common,name="dtw")
    if(nn):
        common.append("%s/%s/feats" % (dir_path,nn))
    binary="%s/%s/feats" % (dir_path,binary)
    return {"common":common,"binary":binary}