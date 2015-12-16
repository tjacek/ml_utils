import dataset
import numpy as np

def unify(in_path1,in_path2,out_path):
    dataset1=dataset.labeled_to_dataset(in_path1)
    dataset2=dataset.labeled_to_dataset(in_path2)
    assert(dataset1.size==dataset2.size)
    new_X=[]
    for i in range(dataset1.size):
    	xa_i=dataset1.X[i]
        xb_i=dataset2.X[i]
        x_full=np.concatenate((xa_i,xb_i))
        new_X.append(x_full)
    	print(x_full.shape)
    new_X=np.array(new_X)
    new_dataset=dataset.LabeledDataset(new_X,dataset1.y)
    csv_text=new_dataset.to_csv()
    text_file = open(out_path, "w")
    text_file.write(csv_text)
    text_file.close()

if __name__ == "__main__":
	path="../af/result/"
	in_path1=path+"dataset"
	in_path2=path+"dataset_hard_full"
	out_path=path+"full_dataset"
	unify(in_path1,in_path2,out_path)