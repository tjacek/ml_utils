import files

def basic_paths(dataset,dir_path,common,binary,name="dtw"):
    paths={}
    if((dataset is None) and (dir_path is None)):
        paths["binary"]=binary
        paths["common"]=files.get_paths(common,name=name)
        return paths
    if( dataset and dir_path ):
        paths["dir_path"]="%s/%s" % (dir_path,dataset)
    elif( dir_path is None):
        paths["dir_path"]=dataset
    else:
        paths["dir_path"]=dir_path
    common="%s/%s" % (paths["dir_path"],common)
    paths["common"]=files.get_paths(common,name=name)
    if(binary):
        paths["binary"]="%s/%s" % (paths["dir_path"],binary)
    else:
        paths["binary"]=None
    return paths 

def get_metrics(result_i):
	acc_i= result_i.get_acc()
	metrics="%.4f,%.4f,%.4f" % result_i.metrics()[:3]
	return "%.4f,%s" % (acc_i,metrics)