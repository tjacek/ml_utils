import exper,exper.voting,exper.persons
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def acc_curve(dict_args):
    quality=exper.persons.quality(**dict_args)
    votes=get_votes(dict_args,clf_type="LR")
    acc=[selected_voting(i+1,quality,votes) 
            for i in range(len(quality)) ]
    #acc=[ clf_select(i+1,dict_args,quality) 
    #        for i in range(len(quality))]
    print(acc)    
    plt.plot(acc)
    plt.ylabel("acc")
    plt.show()

def clf_select(n_clf,args,quality):
    votes=get_votes(args,clf_type="LR")
    return selected_voting(n_clf,quality,votes)
    
def get_votes(args,clf_type="LR"):
    datasets=exper.voting.get_datasets(**args)
    return [exper.predict_labels(data_i,clf_type)  
                for data_i in datasets]

def selected_voting(n_clf,quality,votes):
    print(quality[:n_clf])
    votes=[votes[i] for i in quality[:n_clf]]
    y_true,y_pred=exper.voting.count_votes(votes)
    print(classification_report(y_true, y_pred,digits=4))
    return accuracy_score(y_true,y_pred)