import sys
sys.path.append("..")
import xgboost as xgb
import files,feats,learn,exp

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
    X,y,names=train_i.as_dataset() 
    n_estimators = 10
    max_depth = 10
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=1)
    model.fit(X, y)
    booster = model.get_booster()
    return test_model(model,test_i)

def test_model(model,test):
    from sklearn.metrics import accuracy_score
    X,y,names=test.as_dataset() 
#   dtest = xgb.DMatrix(X, label=y)
    y_pred = model.predict(X)
    return accuracy_score(y,y_pred)

if __name__ == "__main__":
    acc=make_models("../../data/II/common")#,"../../data/II/boost")
    print(acc)