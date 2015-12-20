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
    csv_text=str(new_dataset)
    text_file = open(out_path, "w")
    text_file.write(csv_text)
    text_file.close()

def unify(dataset1,dataset2):
    assert(dataset1.size==dataset2.size)
    new_X=[]
    for i in range(dataset1.size):
    	xa_i=dataset1.X[i]
        xb_i=dataset2.X[i]
        x_full=np.concatenate((xa_i,xb_i))
        new_X.append(x_full)
    new_X=np.array(new_X)
    return dataset.LabeledDataset(new_X,dataset1.y)
    
if __name__ == "__main__":
    in_path="../af/cascade/result/"
    out_path="../af/cascade/full_dataset"
    unify_dir(in_path,out_path)