#import unify,plot,smooth,agum.warp,dataset,synth,extract
import exper.cats,exper.voting,files

def gen_votes(dataset_path,vote_path,n_feats=None,clf_type="LR"):
    hc_path,deep_path=dataset_path+'/hc',dataset_path+'/binary_feats'
    if(not n_feats):
        n_feats=(100,130)
    hc_feats,deep_feats=n_feats
    if(deep_feats):
        s_deep_path=dataset_path+'/s_binary_feats'
        exper.voting.select_feats(deep_path,s_deep_path,n_feats=deep_feats)
        deep_path=s_deep_path
    args={"hc_path":hc_path,"deep_paths":deep_path,'n_feats':(hc_feats,0)}
    files.make_dir(vote_path)
    exper.cats.make_votes(args,vote_path+'/votes',clf_type=clf_type)

def img_dataset(in_path,use_agum=False):
    raw_ts=unify.read(in_path)
    if(use_agum):
        smooth_ts=agum.warp.warp_agum(raw_ts)
    else:
        smooth_ts=raw_ts(smooth.SplineUpsampling())
    dataset.as_imgs(smooth_ts)

def extrac_feats(in_path,out_path):
    raw_ts=unify.read(in_path)
    stats=extract.get_kurt_stats()
    stat_feats=raw_ts.to_feats(stats)
    stat_feats.save(out_path)

def show_smoothing(in_path):
    raw_ts=unify.read(in_path)
    plot.plot_by_feat(raw_ts)
    smooth_ts=raw_ts(smooth.Fourrier())
    plot.plot_by_feat(smooth_ts)

def make_agum(in_path):
    raw_ts=unify.read(in_path)
    agum_ts=agum.warp.warp_agum(raw_ts)
    agum_ts.save()