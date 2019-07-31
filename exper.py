import feats,extract,unify,learn,smooth
from sklearn.metrics import classification_report

def exper_single(in_path,clf_type="SVC",n_select=None):
    feat_dataset=feats.read(in_path)
    feat_dataset.norm()
    y_pred,y_true=predict_labels(feat_dataset,clf_type,n_select)
    print(classification_report(y_true, y_pred,digits=4))

def predict_labels(feat_dataset,clf_type="LR",n_select=None):
    if(n_select):
        feat_dataset.reduce(n_select)
    train,test=split_data(feat_dataset)
    print("Number of features:%d" % train.X.shape[1])
    clf=learn.get_cls(clf_type)    
    clf.fit(train.X,train.get_labels())
    return clf.predict(test.X),test.get_labels()

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

def gen_feats(seq_path,out_path,extractor=None,transform=None):
    ts_dataset=unify.read(seq_path)
    ts_dataset.normalize()
    if(transform):
        ts_dataset=ts_dataset(transform)
    if(not extract):
        extractor=extract.basic_stats
    feat_dataset=ts_dataset.to_feats(extractor)
    feat_dataset.save(out_path)

if __name__ == "__main__":
    #gen_feats("mra","datasets/noise.txt",extract.NoiseCorl())
    exper_single("datasets/exp/","SVC")