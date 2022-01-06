import sys
sys.path.append("..")
sys.path.append("../pref")
import pickle
import ens,pref,pref.optim_algs,pref.evol_score

def simple_exp(votes_path:str):
    pref_dict=get_pref(votes_path)
    result=pref.election(pref_dict.keys(),None,pref_dict)
    result.report()

def evol_exp(train_path:str,test_path:str):
    pref_train=get_pref(train_path)
    pref_test=get_pref(test_path)
    alg_optim=pref.optim_algs.GenAlg(init_type='borda')
    loss_fun=pref.evol_score.AucLoss

    loss_fun=loss_fun(pref_train) 
    n_cand=pref_train.n_cand()
    score=alg_optim(loss_fun,n_cand)
    print(score)
    result=pref.eval_score(score,pref_test)
    result.report()

def get_pref(votes_path:str):
    with open(votes_path, 'rb') as votes_file:
        votes=pickle.load(votes_file)	
        return pref.to_pref(votes.results)


#def eff_evol_score(score,pref_test):

#simple_exp(votes_path:str)
evol_exp("penglung/validate","penglung/bag_votes")