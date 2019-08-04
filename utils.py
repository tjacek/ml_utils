import unify,plot,smooth,agum.warp

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

dataset=make_agum("../ActionClassifier/mra/seq")#,out_path='agum')