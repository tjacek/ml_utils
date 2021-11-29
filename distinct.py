import numpy as np
import seqs,learn

def find_discrm(in_path):
    seq_dict=seqs.read_seqs(in_path)
    seq_dict=seq_dict.split()[0] 
    for p in range(1,6):
        selector=lambda name_i:name_i.get_person()!=(2*p-1)
        result_p=train_model(seq_dict,selector)
        result_p.report()
#    train,test=seq_dict.split(selector)
#    print(len(train))

def train_model(seq_dict,selector,clf_type="LR"):
    train,test=seq_dict.split(selector)
    train_data=to_dataset(train)
    test_data=to_dataset(test)
    model=learn.get_cls(clf_type)
    print(train_data[0].shape)
    model.fit(train_data[0],train_data[1])
    y_pred=model.predict(test_data[0])
    return learn.Result(test_data[1],y_pred,test_data[2])

def to_dataset(seq_dict):
    X,y,names=[],[],list(seq_dict.keys())
    for name_i in names:	
        for frame_j in seq_dict[name_i]:
        	X.append(frame_j)
        	y.append(name_i.get_cat())
    return np.array(X),y,names

in_path="../cc2/ae_seqs"
find_discrm(in_path)