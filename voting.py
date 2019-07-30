import files,feats

def voting(hc_path,deep_paths):
    hc_feats,full_feats=feats.read(hc_path),[]
    for path_i in files.top_files(deep_paths):
        deep_feats_i=feats.read(path_i)
        full_feats.append( hc_feats +deep_feats_i)
    return full_feats

voting('datasets/exp','deep')