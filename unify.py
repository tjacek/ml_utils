import dataset
import numpy as np
import os

def unify_dir(in_path,out_path):
    all_files=os.listdir(in_path)
    def is_file(f):
        return  os.path.isfile(os.path.join(in_path,f))
    dataset_paths=filter(is_file ,all_files)
    dataset_paths=[in_path+ path for path in dataset_paths]
    print(dataset_paths)
    new_dataset=dataset.annotated_to_dataset(dataset_paths[0])
    for i in range(1,len(dataset_paths)):
        partial_dataset=dataset.annotated_to_dataset(dataset_paths[i])
        new_dataset=unify(new_dataset,partial_dataset)
    save_dataset(new_dataset,out_path)
    return new_dataset

def unify(dataset1,dataset2):
    assert(dataset1.size==dataset2.size)
    new_X=[]
    for i in range(dataset1.size):
    	xa_i=dataset1.X[i]
        xb_i=dataset2.X[i]
        x_full=np.concatenate((xa_i,xb_i))
        new_X.append(x_full)
    new_X=np.array(new_X)
    return dataset.AnnotatedDataset(new_X,dataset1.y,dataset1.anno)

def split(dataset0,out_path):
    even=np.array([ (k% 2)==1 for k in dataset0.anno],dtype=bool)
    odd=np.array([ (k% 2)==0 for k in dataset0.anno],dtype=bool)
    dataset1=select_data(dataset0,even)
    dataset2=select_data(dataset0,odd)
    save_dataset(dataset1,out_path+"_train")
    save_dataset(dataset2,out_path+"_test")

def select_data(dataset0,index):
    new_X=np.array(dataset0.X[index])
    new_y=np.array(dataset0.y).T
    new_y=new_y[index]
    #new_anno=np.array(dataset0.anno)[index]
    return dataset.LabeledDataset(new_X,new_y)#dataset.AnnotatedDataset(new_X,new_y,new_anno)

def save_dataset(dataset,out_path):
    csv_text=str(dataset)
    text_file = open(out_path, "w")
    text_file.write(csv_text)
    text_file.close()

if __name__ == "__main__":
    in_path="../af/cascade/result/"
    out_path="../af/cascade/full_dataset"
    dataset0=unify_dir(in_path,out_path)
    split(dataset0,out_path)