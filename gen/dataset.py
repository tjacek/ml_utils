import sys
sys.path.append("..")
import files,feats,ens,learn

def eff_voting(paths,clf="LR"):
    all_results=[]
#    all_results.append(learn.train_model(feats.read(paths[0])[0]))
    for i,data_i in enumerate(read_multi(paths)):
        all_results.append(learn.train_model(data_i,clf_type=clf))
        print(i)
    return ens.Votes(all_results)

def read_dataset(paths):
    import gc
    common_path,binary_path=paths
    if(common_path):
        common=feats.read(common_path)[0]
    if(binary_path is None):
        yield common
        return None
    for deep_path_i in files.top_files(binary_path):
        gc.collect()
        data_i=feats.read(deep_path_i)[0]
        if(common_path):
            data_i=common+data_i
        yield data_i

def read_multi(paths):
    common_paths,binary_path=paths
    if(type(common_paths)==str):
        common_paths=[common_paths]
    for common_i in common_paths:
        for data_j in read_dataset((common_i,binary_path)):
            yield data_j

if __name__ == "__main__":    
    paths=('penglung3/common','penglung3/binary')
#    paths=(None,'forest/binary')
    final_votes=eff_voting(paths,clf="Tree")
    result=final_votes.voting()
    result.report()
    print(result.get_acc())
#    final_votes.save("penglung/votes")