import sys
sys.path.append("..")
import learn,ens,files,dataset,convert,bag

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

def bag_validate(dataset,out_path):
    train,test=dataset.split()
    results=[]
    for data_i in bag.resample_dataset(dataset):
        model_i=learn.train_model(data_i,clf_type="LR",model_only=True)
        y_pred=model_i.predict_proba(train.get_X())
        result_i= learn.Result(train.get_labels(),y_pred,train.names())    
        results.append(result_i)
    validate_votes=ens.Votes(results)
    files.save(out_path,validate_votes)

if __name__ == "__main__":    
    dataset=convert.txt_dataset("penglung/raw.data")
#    dataset_gen=bag.resample_dataset(data)
    bag_validate(dataset,"penglung/validate")
#    make_validate(dataset_gen,"penglung/validate")