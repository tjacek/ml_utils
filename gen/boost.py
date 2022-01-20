import sys
sys.path.append("..")
import xgboost as xgb
import numpy as np
from sklearn import ensemble
import files,feats,learn,exp,ens

@exp.save_results
@files.dir_function(args=1,recreate=False)
def make_datasets(in_path):
    print(in_path)
    data_i=feats.read(in_path)[0]
    train_i,test_i=data_i.split()
    X,y,names=train_i.as_dataset() 
    dtrain = xgb.DMatrix(X, label=y)
    X,y,names=test_i.as_dataset() 
    dtest = xgb.DMatrix(X, label=y)
    num_round=10
    param={'max_depth': 2,'eval_metric':'auc',
    'objective':'multi:softmax','num_class':data_i.names().n_cats()}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    y_pred = bst.predict(dtest)
    result_i=learn.Result(y,y_pred,names)
    return (in_path.split('/')[-1],result_i)

@files.dir_function(args=1,recreate=False)
def make_models(in_path):#,out_path):
    print(in_path)
    data_i=feats.read(in_path)[0]
    train_i,test_i=data_i.split()
    X_train,y_train,names=train_i.as_dataset() 
    n_cats=data_i.n_cats()
    clf_i = ensemble.GradientBoostingClassifier(n_estimators=10,
        max_depth=3)
    clf_i.fit(X_train,y_train)
    X_test,y_test,names_test=test_i.as_dataset()
    print(n_cats)
    print(len(y_test))
    results=[]
    for est_j in clf_i.estimators_:
        y_raw=np.array([tree.predict(X_test)
                          for tree in est_j])        
        if(y_raw.shape[0]>1):
            y_pred=np.argmax(y_raw,axis=0)
        else:
            y_pred= (y_raw<0).astype(int)
            y_pred= y_pred.ravel()
        result_j=learn.Result(y_test,y_pred,names_test)
        results.append(result_j)
    votes_i=ens.Votes(results)
    result=votes_i.voting()
    result.report()
#    pred=clf_i._raw_predict(X_test)
#    print(pred.shape)

#    raise Exception(pred.shape)
#    raise Exception(dir(clf_i))
#    y_pred=clf_i.predict(X_test)
#    result_i=learn.Result(y_test,y_pred,names_test)
#    return result_i


def test_model(model,test):
    from sklearn.metrics import accuracy_score
    X,y,names=test.as_dataset() 
#   dtest = xgb.DMatrix(X, label=y)
    y_pred = model.predict(X)
    return accuracy_score(y,y_pred)


if __name__ == "__main__":
    results=make_models("../../data/II/common")#,"models")
#    prepare_votes("models")
#    for result_i in results:
#        print(result_i.get_acc())