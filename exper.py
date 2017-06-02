import dataset
import eval

def experiment_full(in_path,in_path2):
    data=dataset.read_and_unify(in_path,in_path2)
    even_data=dataset.select_person(data,i=0)
    odd_data=dataset.select_person(data,i=1)
    eval.determistic_eval(odd_data,even_data,svm=False)

def experiment_restricted(in_path,in_path2,cats=[],from_zero=False):
    data=dataset.read_and_unify(in_path,in_path2)
    if(from_zero):
        cats=[(cat_i-1) for cat_i in cats]
    r_data=dataset.select_category(data,cats)
    even_data=dataset.select_person(r_data,i=0)
    odd_data=dataset.select_person(r_data,i=1)
    eval.determistic_eval(odd_data,even_data,svm=False)


if __name__ == "__main__":
    in_path='../reps/dtw_feat/dataset.txt'
    in_path2= '../ultimate3/simple/dataset.txt'#'../reps/dtw_feat/simple3/dataset.txt'
    A1=[2,3,5,6,10,13,18,20]
    A2=[1,4,7,8,9,11,12,14]
    A3=[6,14,15,16,17,18,19,20]
    experiment_restricted(in_path,in_path2,A3,from_zero=True)