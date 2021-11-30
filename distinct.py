import numpy as np
import shutil
import seqs,learn,files

def find_discrm(in_path,out_path):
    seq_dict=seqs.read_seqs(in_path)
    seq_dict=seq_dict.split()[0] 
    discrm_frames=[]
    for p in range(1,6):
        selector=lambda name_i:name_i.get_person()!=(2*p-1)
        result_p=train_model(seq_dict,selector)
        discrm_frames+=result_p.get_correct()
    print(len(discrm_frames))
    return discrm_frames
#    save_discrm(discrm_frames,out_path)

def save_discrm(in_path,discrm_frames,out_path,n_cats=9):
    files.make_dir(out_path)
    for i in range(n_cats):
    	files.make_dir("%s/%d" % (out_path,i+1))
    path_dict=files.get_path_dict(in_path)
    for name_i,j in discrm_frames:
        in_ij=path_dict[name_i][j]
        out_ij="%s/%d/%s_%d" % (out_path,name_i.get_cat()+1,name_i,j)
        print(in_ij)
        print(out_ij)       
        shutil.copyfile(in_ij,out_ij)

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
    names=list(seq_dict.keys())
    X,y,frame_names=[],[],[]
    for name_i in names:	
        for j,frame_j in enumerate(seq_dict[name_i]):
        	frame_names.append((name_i,j))
        	X.append(frame_j)
        	y.append(name_i.get_cat())
    return np.array(X),y,frame_names

in_path="../cc2/ae_seqs"
out_path="disc"
discrm_frames=find_discrm(in_path,out_path)
save_discrm("../cc2/final",discrm_frames,out_path,n_cats=9)