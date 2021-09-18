import files,ens

class EnsembleExp(object):
    def __init__(self,ensemble=None):
        if(ensemble is None):
            ensemble=ens.Ensemble()
        self.ensemble=ensemble
        
    def __call__(self,input_dict):
        lines=[]
        for desc_i,path_i in simple_gen(input_dict):
            if(type(path_i)==tuple):
                path_i={"common":path_i[0],"binary":path_i[1]}
            print(path_i)
            result_i=self.ensemble(path_i)[0]
            line_i="%s,%s" % (desc_i,get_metrics(result_i))
            lines.append(line_i)
        print(lines)
        return lines

def simple_gen(input_dict):
    print(input_dict)
    common,binary=input_dict
    for common_i in common:
        desc_i=common_i.split("/")[-1]
        yield desc_i,(common_i,binary)

def get_metrics(result_i):
	acc_i= result_i.get_acc()
	metrics="%.4f,%.4f,%.4f" % result_i.metrics()[:3]
	return "%.4f,%s" % (acc_i,metrics)

def get_out_path(in_path,name):
    dir_path="_".join(in_path.split("/")[:-1])
    return "%s/%s" % (in_path,name)

def fill_template(template,elements):
    tuples=[]  
    for element_i in elements:
        if(tuples):
            if(type(element_i)==str):
                for tuple_i in tuples:
                    tuple_i.append(element_i)
            else:
                tuples=[ tuple_i+[element_j] 
                            for tuple_i in tuples
                                for element_j in element_i]
        else:
            if(type(element_i)==str):
                element_i=[element_i]
            tuples.append(element_i)
    return [ template % tuple(tuple_i) for tuple_i in tuples]