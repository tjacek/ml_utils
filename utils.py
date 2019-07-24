import unify,plot,smooth

def show_smoothing(in_path):
    raw_ts=unify.read(in_path)
    plot.plot_by_feat(raw_ts)
    #smooth_ts=raw_ts(smooth.Gauss())
    smooth_ts=raw_ts(smooth.Fourrier())
    plot.plot_by_feat(smooth_ts)

dataset=show_smoothing("mra")