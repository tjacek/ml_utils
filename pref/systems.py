import numpy as np

def majority(name_i,pref_dict):
    first=pref_dict.get_rank(name_i,0)
    unique, counts = np.unique(first, return_counts=True)
    winner= unique[np.argmax(counts)]
    return winner

def borda_count(name_i,pref_dict):
    n_cand=pref_dict.n_cand()
    score=[n_cand-j for j in range(n_cand)]
    return score_rule(name_i,pref_dict, n_cand,score)

def score_rule(name_i,pref_dict, n_cand,score):
    count=np.zeros((n_cand,))
    for j in range(n_cand):
        for vote_k in pref_dict.get_rank(name_i,j):
            count[vote_k]+=score[j]
    return np.argmax(count)

def bucklin(name_i,pref_dict):
    n_cand=pref_dict.n_cand()
    count=np.zeros((n_cand,))
    threshold= np.floor(pref_dict.n_votes()/2)
    for j in range(n_cand):
        for vote_k in pref_dict.get_rank(name_i,j):
            count[vote_k]+=1
        if(np.amax(count)>threshold):
            return np.argmax(count)
    raise Exception("error")

def coombs(name_i,pref_dict):
    counters=pref_dict.as_counters(name_i)
    while(counters):
        threshold=np.floor(np.sum(counters[0])/2)
        if(threshold<np.amax(counters[0])):
            return np.argmax(counters[0])
        if(np.amax(counters[-1])==0):
            counters.pop()
        worst=np.argmax(counters[-1])
        for counter_j in counters:
            counter_j[worst]=0
        if(np.amax(counters[0])==0):
            del counters[0]
        print(len(counters))
    raise Exception(counters)

def copeland_method(name_i,pref_dict):
    v=pref_dict[name_i]
    threshold=pref_dict.threshold()
    n_cand= pref_dict.n_cand()
    def helper(pair_winners):
        score=np.sum(pair_winners)
        if(score> threshold):
            return 1.0
        elif(score==threshold):
            return 0.5
        else:
            return 0.0
    cope=[[helper(pref_dict.pair_winner(name_i,a,b))
            for a in range(n_cand)]
                for b in range(n_cand)]
    cope=np.array(cope)
    copeland_score=np.sum(cope,axis=1)
    return (np.argmax( copeland_score))