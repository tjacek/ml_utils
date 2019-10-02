#from sklearn.linear_model import LogisticRegression
import exper,exper.voting
import files,feats

def votes_dist(args,out_path,split=True):#,clf_type="LR",out_path=None):    
    datasets=exper.voting.get_datasets(**args)
    files.make_dir(out_path)
    votes=[]
    for i,data_i in enumerate( datasets):
        model_i=exper.make_model(data_i)
        if(split):
            train,test=exper.split_data(data_i)
            X,info=test.X,test.info
        else:
            X,info=data_i.X,data_i.info
        dist_i=model_i.predict_proba(X)
        vote_i=feats.FeatureSet(dist_i,info)
        if(out_path):
            out_i=out_path+'/votes'+str(i)
            vote_i.save(out_i)
        votes.append(vote_i)
    return votes    