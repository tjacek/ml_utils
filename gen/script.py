import sys
sys.path.append("..")
import simple,convert,dataset,files
import voting,validate

datset_path="penglung/raw.data"
dir_path="penglung3"
data=convert.txt_dataset(datset_path)
#simple.make_dataset(data,dir_path,n_epochs=150)
paths=([None,f"{dir_path}/common"],f"{dir_path}/binary")
clf="Tree"

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

    return {f"single,{clf}":result_single,
            f"mult,{clf}":result_multi,
            f"ovp,{clf}":result_ovp}

def save_results(result_dict,out_path):
    import exp
    lines=[]
    for name_i,result_i in result_dict.items():	
        line_i=exp.get_metrics(result_i)
        lines.append(f"{name_i},{line_i}")
    exp.save_lines(lines,out_path)

result_dict=ovp_exp(paths,opv_path,clf)
for result_i in result_dict.values():
	print(result_i.get_acc())

#raise Exception(f"{opv_path}/{clf}.csv")
save_results(result_dict,f"{opv_path}/{clf}.csv")