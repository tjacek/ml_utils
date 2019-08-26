import numpy as np
import voting,filtr

def quality(**args):
    datasets=voting.get_datasets(**args)
    pred_acc(datasets[0])

def pred_acc(data_i):
    train,test=filtr.split(data_i.info)
    train_data=filtr.filtered_dict(train,data_i)
    by_person=samples_by_person(train)
    def one_out(i):
        one,rest=by_person[i],[]
        for person_j,names_j in by_person.items():
            if(person_j!=i):
                rest+=names_j
        return one,rest
    for person_i in by_person.keys():
        one,rest=one_out(person_i)
        print(len(one))
        print(len(one)+len(rest))

def samples_by_person(train):
    persons_dict=get_person( train)
    person_ids= list(np.unique(persons_dict.values()))
    by_person={person_i:
                    [name_j for name_j,person_j in persons_dict.items()
                        if(person_j==person_i)]
                            for person_i in person_ids}
    return by_person

def get_person(names):
    return { name_i:name_i.split('_')[1]  for name_i in names}