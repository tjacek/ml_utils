import unify,plot,smooth,agum.warp,dataset,synth,extract

def synth_dataset(out_path):
    means=[-2.0,-1.0,0.0,1.0,2.0]
    stds=[1.0,2.0,3.0,4.0]
    skew=[-1.0,0.0,1.0]
    norm_dist=synth.skew_gen(means,stds,skew,n_cats=20,
                    n_size=250,ts_len=128,n_feats=12)
    print(len(norm_dist))
    dataset.as_imgs(norm_dist,out_path)

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
    #smooth_ts=raw_ts(smooth.Gauss())
    smooth_ts=raw_ts(smooth.Fourrier())
    plot.plot_by_feat(smooth_ts)

def make_agum(in_path):
    raw_ts=unify.read(in_path)
    agum_ts=agum.warp.warp_agum(raw_ts)#smooth_agum(raw_ts)
    agum_ts.save()

synth_dataset("../pretrain")
#extrac_feats('mra_','exp2/kurt.txt')