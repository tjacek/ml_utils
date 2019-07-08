import feats,extract,unify,learn
from sklearn.metrics import classification_report

def exper_single(in_path,clf_type="SVC"):
    feat_dataset=feats.read(in_path)
    feat_dataset.norm()
    train,test=split_data(feat_dataset)
    clf=learn.get_cls(clf_type)    
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

def gen_feats(seq_path,out_path,extractor=None):
    ts_dataset=unify.read(seq_path)
    if(not extract):
        extractor=extract.basic_stats
    feat_dataset=ts_dataset.to_feats(extractor)
    feat_dataset.save(out_path)

if __name__ == "__main__":
    #gen_feats("mra","nonlin.txt",extract.non_linear)
    exper_single("datasets","SVC")