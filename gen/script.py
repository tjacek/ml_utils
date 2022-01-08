import sys
sys.path.append("..")
import simple,convert,dataset,files
import voting,validate

datset_path="penglung/raw.data"
dir_path="penglung3"
data=convert.txt_dataset(datset_path)
#simple.make_dataset(data,dir_path,n_epochs=150)
paths=([None,f"{dir_path}/common"],f"{dir_path}/binary")
clf="LR"

opv_path=f"{dir_path}/{clf}"

def ovp_exp(paths,opv_path,clf):
    files.make_dir(opv_path)
    dataset_gen=dataset.read_multi(paths)
    train_path,test_path=f'{opv_path}/validate',f'{opv_path}/test'
    validate.make_validate(dataset_gen,train_path,clf=clf)
    voting_multi=dataset.eff_voting(paths,clf=clf)
    voting_multi.save(test_path)
    result_multi=voting_multi.voting()
    result_ovp=voting.evol_exp(train_path,test_path)

    common=paths[0]
    if(type(common)==list):
    	common=[ common_i for common_i in common
    	             if(common_i)]
    result_single=dataset.eff_voting((common,None),clf=clf).voting()

    return [result_single,result_multi,result_ovp]

results=ovp_exp(paths,opv_path,clf)
for result_i in results:
	print(result_i.get_acc())