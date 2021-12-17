import ens

def validate(paths,ensemble):
    datasets=ensemble.get_datasets(paths)
    raise Exception(len(datasets))

def get_pref_dict(paths,ensemble):
    result,votes=ensemble(paths)
    pref_dict=pref.to_pref(votes.results)
    return pref_dict
