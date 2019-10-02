#from sklearn.linear_model import LogisticRegression
import exper,exper.voting
import files,feats

def votes_dist(args,out_path):#,clf_type="LR",out_path=None):    
    datasets=exper.voting.get_datasets(**args)
    files.make_dir(out_path)
    for i,data_i in enumerate( datasets):
        model_i=exper.make_model(data_i)
        dist_i=model_i.predict_proba(data_i.X)
        vote_dist=feats.FeatureSet(dist_i,data_i.info)
        vote_dist.save(out_path+'/votes'+str(i))