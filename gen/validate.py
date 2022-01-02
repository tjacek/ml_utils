import sys
sys.path.append("..")
import learn,ens,files,dataset

def make_validate(dataset_gen,out_path):
    results=learn.person_votes(dataset_gen,
    	get_partial=enum_partial,clf="LR",n_split=5)
    validate_votes=ens.Votes(results)
    files.save(out_path,validate_votes)

def enum_partial(train,n_split,clf):
    for p in range(n_split):
        def selector_p(name):
            return (int(name.split("_")[-1])%n_split)!=p
        validate,validate_test=train.split(selector_p)
        yield learn.make_result(validate,validate_test,clf)

if __name__ == "__main__":    
    paths=('forest/common','forest/binary')
    make_validate(dataset.read_dataset(paths),"validate_forest")