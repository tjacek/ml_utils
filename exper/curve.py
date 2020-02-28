import matplotlib.pyplot as plt
import learn,feats,exper.cats

def acc_curve(vote_path,ord,binary=False,show=True):
    votes=selected_votes(vote_path,ord,binary)
    n_clf=len(votes)
    results=[exper.cats.voting(votes[:(i+1)],None) 
                for i in range(n_clf)]
    acc=[ learn.compute_score(result_i[0],result_i[1],False)[0] 
            for result_i in results]
    if(show):
        show_curve(acc,len(results),vote_path,binary)
    return acc

def selected_votes(vote_path,ord,binary=False):
    votes=feats.read_list(vote_path)
    if(binary):
        votes=[binarize(vote_i) for vote_i in votes]
    return [ votes[k] for k in ord]
    
def show_curve(acc,n_clf,vote_path,binary):
    title="_".join(vote_path.split('/'))
    title=title.replace(".._","").replace("votes","")
    voting_type= "HARD" if(binary) else "SOFT"
    title+=voting_type
    plt.title(title)
    plt.grid(True)
    plt.xlabel('number of classifiers')
    plt.ylabel('accuracy')
    plt.plot(range(1,n_clf+1), acc, color='red')
    plt.show()
    return acc