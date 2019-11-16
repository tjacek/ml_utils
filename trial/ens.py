import exper,exper.voting
import files,feats

def cls_votes(in_path,out_path):
    feat_dataset=feats.read(in_path)
    files.make_dir(out_path)
    for cls_i in ['SVC','LR']:
        model_i=exper.make_model(feat_dataset,cls_i)
        probs=model_i.predict_proba(feat_dataset.X)
        feats_i=feats.FeatureSet(probs,feat_dataset.info) 
        feats_i.save(out_path+"/"+cls_i)

def mixed_cls(args,out_path):
    datasets=exper.voting.get_datasets(**args)
    files.make_dir(out_path)
    clf_types=['SVC','LR']
    for i,data_i in enumerate(datasets):
        clf_i=clf_types[i%2]
        model_i=exper.make_model(data_i,clf_i)
        probs=model_i.predict_proba(data_i.X)
        feats_i=feats.FeatureSet(probs,data_i.info) 
        feats_i.save(out_path+"/nn"+str(i))