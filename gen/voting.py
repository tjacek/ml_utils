import sys
sys.path.append("..")
sys.path.append("../pref")
import pickle
import traceback
import ens,pref,pref.optim_algs,exp
import pref,pref.loss,files#.evol_score

def simple_exp(votes_path:str):
    pref_dict=get_pref(votes_path)
    result=pref.election(pref_dict.keys(),None,pref_dict)
    result.report()

def evol_exp(train_path:str,test_path:str):
    pref_train=get_pref(train_path)
    pref_test=get_pref(test_path)
    alg_optim=pref.optim_algs.GenAlg(init_type='borda')
    loss_fun=pref.loss.AucLoss

    loss_fun=loss_fun(pref_train) 
    n_cand=pref_train.n_cand()
    score=alg_optim(loss_fun,n_cand)
    result=pref.eval_score(score,pref_test)
    print(score)
    result.report()
    return result

@exp.save_results
def ens_evol(train,test):
    base,opv,names=[],[],[]
    paths=zip(files.top_files(train),files.top_files(test))
    for path_i in paths:
        print(len(path_i))
        print(path_i)
#        try:
        base.append(get_pref(path_i[0],True))
        opv.append(evol_exp(*path_i))
        desc_i=path_i[0].split("/")[-1]
        names.append(f'{desc_i},{base[-1].n_cats()}')
#        except Exception as e:
#            traceback.print_exc()
    result_dict={}
    for i,(base_i,opv_i) in enumerate(zip(base,opv)):
        result_dict[f'{names[i]},base']=base_i
        result_dict[f'{names[i]},opv']=opv_i
    print(result_dict)
    return result_dict

def get_pref(votes_path:str,raw_votes=False):
    with open(votes_path, 'rb') as votes_file:
        votes=pickle.load(votes_file)
        if(len(votes)==0):
            raise Exception(votes_path)
        if(raw_votes):
            return votes.voting()
        return pref.to_pref(votes.results)

if __name__ == "__main__":    
#    simple_exp("penglung/votes")
    ens_evol("binary/full.csv","binary/valid","binary/test")