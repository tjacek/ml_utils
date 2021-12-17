import ens,learn,pref

def validate(paths,ensemble):
    datasets=ensemble.get_datasets(paths)
    results= learn.validation_votes(datasets,clf="LR")
    train_dict=pref.to_pref(results)
    result,votes=ensemble(paths)
    test_dict=pref.to_pref(votes.results)
    return train_dict,test_dict
#    raise Exception(len(pref_dict))

def person_dict(paths,ensemble):
    datasets=ensemble.get_datasets(paths)
    results= learn.person_votes(datasets,clf="LR")
    train_dict=pref.to_pref(results)
    result,votes=ensemble(paths)
    test_dict=pref.to_pref(votes.results)
    return train_dict,test_dict

def pref_dict(paths,ensemble):
    result,votes=ensemble(paths)
    pref_dict=pref.to_pref(votes.results)
    return pref_dict,pref_dict
