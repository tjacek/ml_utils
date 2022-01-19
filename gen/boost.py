import sys
sys.path.append("..")
import xgboost as xgb
import files,feats
from sklearn.metrics import accuracy_score

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
    ypred = bst.predict(dtest)
    acc_i=accuracy_score(y,ypred) 
    return acc_i

if __name__ == "__main__":
    acc=make_datasets("../../data/II/common")#,"../../data/II/boost")
    print(acc)