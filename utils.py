import unify,plot,smooth,agum.warp,dataset,synth

def synth_dataset(out_path):
    means=[-1.0,0.0,1.0]
    stds=[1.0,2.0,3.0,4.0]
    norm_dist=synth.normal_gen(means,stds,n_size=250,
                            ts_len=128,n_feats=12)
    dataset.as_imgs(norm_dist,out_path)

def img_dataset(in_path,use_agum=False):
    raw_ts=unify.read(in_path)
    if(use_agum):
        smooth_ts=agum.warp.warp_agum(raw_ts)
    else:
        smooth_ts=raw_ts(smooth.SplineUpsampling())
    dataset.as_imgs(smooth_ts)

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

synth_dataset("../pretrain")#,out_path='agum')