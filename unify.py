import feats,files 

def unify_dataset(path,out_path):
    dataset= feats.read_unified(files.top_files(path))
    print(dataset.dim())
    dataset.save(out_path)

in_path="../conv_frames/test/simple_feats"
unify_dataset(in_path,"3DHOI_simple")