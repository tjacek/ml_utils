import sys
sys.path.append("..")
sys.path.append("../optim")
import numpy as np
import optim.diff_evol,optim

def select_clfs(paths,read_type=None):
    optim_alg=optim.diff_evol.OptimizeWeights(optim.gasen.Corl,
        maxiter=10,read=read_type)
    datasets=optim_alg.read(*paths)
    weights=optim_alg.find_weights(datasets)
    print(weights)
    clfs=np.argsort(weights)
    print(clfs)

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]#path % "shapelets"]
    binary=path % "ens/splitII/"
    paths=(common,binary)
    select_clfs(paths,read_type=None)
