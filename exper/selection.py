import exper,exper.voting,exper.persons
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def acc_curve(dict_args):
    quality=exper.persons.quality(**dict_args)
    acc=[ clf_select(i+1,dict_args,quality) 
            for i in range(len(quality)-1)]
    print(acc)    
    plt.plot(acc)
    plt.ylabel("acc")
    plt.show()

def clf_select(n_clf,args,quality):
    print(quality)
    datasets=exper.voting.get_datasets(**args)
    datasets=[datasets[i] for i in quality[:n_clf]]
    print("Used datasets")
    print(quality[:n_clf])
    votes=[exper.predict_labels(data_i,clf_type="LR")  
            for data_i in datasets]
    y_true,y_pred=exper.voting.voting(votes)
    print(classification_report(y_true, y_pred,digits=4))
    return accuracy_score(y_true,y_pred)