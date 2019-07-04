import feats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#from sets import Set

def exper_single(in_path):
    feat_dataset=feats.read(in_path)
    feat_dataset.norm()
    train,test=split_data(feat_dataset)
    clf=LogisticRegression()
    clf.fit(train.X,train.get_labels())
    y_pred=clf.predict(test.X)
    y_true=test.get_labels()
    print(classification_report(y_true, y_pred,digits=4))

def split_data(feat_dataset):
    train,test={},{}
    feat_dict=feat_dataset.to_dict()
    for name_i,data_i in feat_dict.items():
        if(person_selector(name_i)):
            train[name_i]=data_i
        else:
            test[name_i]=data_i
    return feats.from_dict(train),feats.from_dict(test)

def person_selector(name_i):
    return (int(name_i.split('_')[1])%2)==1 

if __name__ == "__main__":
    exper_single("btf.txt")