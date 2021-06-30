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

paths=exp.fill_template("../%s/%s/n_feats",["MHAD",["corl","max_z"]])
exp1(*paths)