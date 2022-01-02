import sys
sys.path.append("..")
sys.path.append("../pref")
import pickle
import ens,pref

def pref_exp(votes_path:str):
    with open(votes_path, 'rb') as votes_file:
        votes=pickle.load(votes_file)	
        pref_dict= pref.to_pref(votes.results)
        print(type(pref_dict))
        result=pref.election(pref_dict.keys(),None,pref_dict)
        result.report()

pref_exp("forest_votes")