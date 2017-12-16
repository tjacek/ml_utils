import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def show_result(y_pred,y_true,conf=True):
    print(classification_report(y_true, y_pred,digits=4))
    if(conf):
       show_confusion(confusion_matrix(y_true, y_pred))
    
def show_confusion(cf_matrix):
    cf_matrix=pd.DataFrame(cf_matrix,index=range(cf_matrix.shape[0]))
    print(cf_matrix)

def show_errors(y_pred,y_true,dataset):
    errors=get_errors(y_pred,y_true)
    persons=dataset.info['persons']
    cats=dataset.info['cats']
    error_names=[ (cats[i],persons[i])
                  for i,error_i in enumerate(errors)
                    if(error_i)]
    return error_names

def get_errors(y_pred,y_true):
    return [ pred_i!=true_i 
	        for pred_i,true_i in zip(y_pred,y_true)]
