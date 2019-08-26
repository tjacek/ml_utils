import numpy as np
import voting,filtr

def quality(**args):
    datasets=voting.get_datasets(**args)
    samples_by_person(datasets[0])

def samples_by_person(data_i):
    train,test=filtr.split(data_i.info)
    train_data=filtr.filtered_dict(train,data_i)
    persons_dict=get_person( train_data)
    person_ids= list(np.unique(persons_dict.values()))
    by_person={person_i:
                    [name_j for name_j,person_j in persons_dict.items()
                        if(person_j==person_i)]
                            for person_i in person_ids}
    print(by_person)

def get_person(names):
    return { name_i:name_i.split('_')[1]  for name_i in names}