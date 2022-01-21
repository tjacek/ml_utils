import sys
sys.path.append("..")
import xgboost as xgb
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
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

@files.dir_function(args=2,recreate=True)
def make_models(in_path,out_path):
    print(in_path)
#    raise Exception(out_path)
    data_i=feats.read(in_path)[0]
    data_i.norm()
    train_i,test_i=data_i.split()
    train_tuple=train_i.as_dataset() 
    n_cats=data_i.n_cats()
    clf_i,best_params=train_boost(train_tuple[0],train_tuple[1])
    test_tuple=test_i.as_dataset()
    
    out_valid,out_test=out_path
    votes_i=get_boost_votes(test_tuple,clf_i)
    votes_i.save(out_valid)

    votes_i=get_boost_votes(train_tuple,clf_i)
    votes_i.save(out_test)

    result=votes_i.voting()
    result.report()
    return (in_path.split("/")[-1],result.get_acc(),best_params)

def get_boost_votes(result_tuple,clf_i):
    X,y_true,names=result_tuple
    results=[]
    for est_j in clf_i.estimators_:
        y_raw=np.array([tree.predict(X)
                          for tree in est_j])        
        if(y_raw.shape[0]>1):
            y_pred=np.argmax(y_raw,axis=0)
        else:
            y_pred= (y_raw<0).astype(int)
            y_pred= y_pred.ravel()
        result_j=learn.Result(y_true,y_pred,names)
        results.append(result_j)
    return ens.Votes(results)

def train_boost(X_train,y_train):
    clf_i = ensemble.GradientBoostingClassifier()#max_depth=3)
    clf_i = GridSearchCV(clf_i,{'max_depth': [2,4,6],
                    'n_estimators': [5,10,15,20]},
                    verbose=1,
                    scoring='neg_log_loss')
    clf_i.fit(X_train,y_train)    
    best_params=clf_i.cv_results_['params'][0]
    return clf_i.best_estimator_,best_params

def test_model(model,test):
    from sklearn.metrics import accuracy_score
    X,y,names=test.as_dataset() 
#   dtest = xgb.DMatrix(X, label=y)
    y_pred = model.predict(X)
    return accuracy_score(y,y_pred)


if __name__ == "__main__":
    results=make_models("../../data/II/common",("valid","test"))#,"models")
#    prepare_votes("models")
    for result_i in results:
        print(result_i)