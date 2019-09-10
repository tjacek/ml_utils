import feats,extract,unify,learn,files
from sklearn.metrics import classification_report,accuracy_score

def exper_single(in_path,clf_type="SVC",n_select=None,norm=True,show=True):
    if(type(in_path)==dict):
        feat_dataset=feats.from_dict(in_path)
    elif(type(in_path)==str):
        feat_dataset=feats.read(in_path)
    else:
        feat_dataset=in_path
    if(norm):
        feat_dataset.norm()
    y_pred,y_true,names=predict_labels(feat_dataset,clf_type,n_select)
    if(show):
        print(classification_report(y_true, y_pred,digits=4))
    return accuracy_score(y_true,y_pred)

def predict_labels(feat_dataset,clf_type="LR",n_select=None):
    if(n_select):
        feat_dataset.reduce(n_select)
    train,test=split_data(feat_dataset)
    if(not len(train)):
        raise Exception("No train data")
    print("Number of features:%d" % train.X.shape[1])
    clf=learn.get_cls(clf_type)
    print(train.X.shape) 
    clf.fit(train.X,train.get_labels())
    return clf.predict(test.X),test.get_labels(),test.info

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