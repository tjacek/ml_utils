import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import feats,learn,files
import exper.cats  

def clf_acc(votes_path,data="test"):
    votes=read_votes(votes_path,data)
    result=[ pred(vote_i) for vote_i in votes]
    return [accuracy_score(*result_i) for result_i in result]

def correct_votes(votes_path,data="train"):
    votes=read_votes(votes_path,data)
    result=[np.diagonal(confusion_matrix(*pred(vote_i))) for vote_i in votes]
    return np.array(result)
#    correct=np.mean(result,axis=0)
#    correct[correct>0.5]=1.0
#    raise Exception(correct)

def read_votes(votes_path,data="test"):
    votes=feats.read_list(votes_path)
    if(data=="test"):
        votes=[ vote_i.split()[1] for vote_i in votes]
    if(data=="train"):
        votes=[ vote_i.split()[0] for vote_i in votes]
    return votes

def pred(data_i):
    y_pred=[np.argmax(x_i) for x_i in data_i.X]
    y_true=data_i.get_labels()
    return y_true,y_pred

def show_votes(votes_path,out_path,data="train"):
    votes=read_votes(votes_path,data=data)
    files.make_dir(out_path)
    for i,vote_i in enumerate(votes):
        out_i="%s/nn%d" % (out_path,i)
        result_i=pred(vote_i)
        learn.show_confusion(result_i,out_i) 

def erorr_vector(data_i):
    if(type(data_i)==list):
        return [ erorr_vector(data_ij) for data_ij in data_i] 
    y_true=data_i.get_labels()
    y_pred=[np.argmax(x_i) for x_i in data_i.X]
    return binary_result(y_true,y_pred)

def binary_result(y_true,y_pred):
    return np.array([float(true_i==pred_i) 
               for true_i,pred_i in zip(y_true,y_pred)])

def to_signal(y):
    y=np.array(y)
    n_cats=max(y)
    return [ (y==i+1).astype(float) for i in range(n_cats)]