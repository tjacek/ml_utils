import numpy as np
import exper,exper.persons,exper.cats
import feats,filtr,learn,files

def voting(in_path,show=True):
    votes=[ feats.read_list(path_i) for path_i in files.top_files(in_path)]
    result=[exper.cats.voting(vote_i,None) for vote_i in votes]
    if(show):
        for result_i in result:
            learn.show_result(*result_i) 
    scores=[ learn.compute_score(result_i[1],result_i[0],as_str=False)
                for result_i in result]
    scores=np.array(scores)
    print(np.mean(scores,axis=0))

def make_votes(args,restr,clf_type,out_path):
    datasets=exper.voting.get_data(args)
    files.make_dir(out_path)
    for i,restr_i in enumerate(restr):
        out_i=out_path+"/restr"+str(i)
        files.make_dir(out_i)
        for j,data_j in enumerate(datasets):
            out_ij=out_i+'/nn'+str(j)
            data_ij=by_cats(data_j,restr_i)
            train,test=data_ij.split()
            pairs=exper.persons.pred_vectors(train,test,clf_type)
            data_ij= feats.from_dict(dict(pairs))
            data_ij.save(out_ij)

def by_cats(feat_set,cat_set):
    cat_set=set(cat_set)
    def cat_selector(name_i):
        cat_i=int(name_i.split('_')[0])
        return (cat_i in cat_set)
    train,test=filtr.split(feat_set.info,cat_selector)
    cat_dict=filtr.filtered_dict(train,feat_set.to_dict())
    cat_dict=filtr.ordered_cats(cat_dict)
    return feats.from_dict(cat_dict)