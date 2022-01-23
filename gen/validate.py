import sys
sys.path.append("..")
import learn,dataset,convert,bag
import files,ens

def enum_partial(train,n_split,clf):
    for p in range(n_split):
        def selector_p(name):
            return (int(name.split("_")[-1])%n_split)!=p
        validate,validate_test=train.split(selector_p)
        yield learn.make_result(validate,validate_test,clf)

@files.dir_function(recreate=False)
def make_validate(in_path,out_path):
    print(in_path)
    print(out_path)
    dataset_gen=dataset.read_dataset(in_path)
#    for data_i in dataset_gen:
#        print(data_i)
    results=learn.person_votes(dataset_gen,
        get_partial=enum_partial,clf="LR",n_split=5)
    validate_votes=ens.Votes(results)
    validate_votes.save(out_path)
    return len(validate_votes)

@files.dir_function(recreate=False)
def make_test(in_path,out_path):
    final_votes=dataset.eff_voting(in_path,clf="LR")
    final_votes.save(out_path)
    return len(final_votes)
    
@files.dir_function(args=1)
def check_votes(votes_path):
    import pickle
    with open(votes_path, 'rb') as votes_file:
        votes=pickle.load(votes_file)
        acc= [result_i.get_acc() for result_i in votes.results]
        print(acc)

if __name__ == "__main__":    
#    paths=([None,'penglung/common'],'penglung/binary')
    in_path=("../../data/II/common2","../../data/II/binary")
    out_path="binary/valid"
#    valid_size=make_validate(in_path,out_path)
#    test_size= make_test(in_path,"binary/test")
    check_votes(out_path)