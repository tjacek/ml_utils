import exper,exper.voting
from sklearn.metrics import classification_report

def clf_select(n_clf,args,quality):
    datasets=exper.voting.get_datasets(**args)
    datasets=[datasets[i] for i in quality[:n_clf]]
    print("Used datasets %d" % len(datasets))
    votes=[exper.predict_labels(data_i,clf_type="LR")  
            for data_i in datasets]
    y_true,y_pred=exper.voting.voting(votes)
    print(classification_report(y_true, y_pred,digits=4))