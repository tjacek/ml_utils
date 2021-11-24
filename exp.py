import files,ens

class MultiEnsembleExp(object):
    def __init__(self,all_ensembles,threshold=0.05):
        self.all_ensembles=all_ensembles
        self.threshold=threshold

    def __call__(self,input_dict,out_path=None):
        lines=[]
        for desc_i,ensemble_i in self.all_ensembles.items():
            result_i,votes_i=ensemble_i(input_dict)
            if(type(votes_i)==ens.Votes):
                n_clf=len(votes_i)
            else:
                n_clf=votes_i[votes_i>self.threshold].shape[0] 
            line_i="%s,%d,%s" % (desc_i,n_clf,get_metrics(result_i))
            lines.append(line_i)
        save_lines(lines,out_path)
        return lines

class EnsembleExp(object):
    def __init__(self,ensemble=None,gen=None):
        if(ensemble is None):
            ensemble=ens.Ensemble()
        if(gen is None):
            gen=simple_gen
        self.ensemble=ensemble
        self.gen=gen
        
    def __call__(self,input_dict,out_path=None):
        lines=[]
        for desc_i,path_i in self.gen(input_dict):
            print(path_i)
            result_i=self.ensemble(path_i)[0]
            line_i="%s,%s" % (desc_i,get_metrics(result_i))
            lines.append(line_i)
        save_lines(lines,out_path)
        return lines

def simple_gen(input_dict):
    print(input_dict)
    common,binary=input_dict
    for common_i in common:
        desc_i=common_i.split("/")[-1]
        yield desc_i,(common_i,binary)

def save_lines(lines,out_path):
    print(lines)
    if(out_path):
        txt="\n".join(lines)
        out_file = open(out_path,"w")
        out_file.write(txt)
        out_file.close()

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