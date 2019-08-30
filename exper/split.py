from sets import Set
import filtr,feats

def by_cats(feat_set,cat_set):
    cat_set=Set(cat_set)
    def cat_selector(name_i):
        cat_i=int(name_i.split('_')[0])
        return (cat_i in cat_set)
    train,test=filtr.split(names,cat_selector)
    cat_dict=filtr.filtered_dict(train,feat_set.to_dict())
    return feats.from_dict(cat_dict)