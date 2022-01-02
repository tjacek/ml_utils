import sys
sys.path.append("..")
import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
import convert,learn,feats,files,ens

class SimpleNN(object):
    def __init__(self,n_hidden=10):
        self.n_hidden=n_hidden
        self.optim=optimizers.RMSprop(learning_rate=0.00001)

    def __call__(self,params):
        model = Sequential()
        model.add(Dense(self.n_hidden, input_dim=params['dims'], activation='relu',name="hidden",
            kernel_regularizer=regularizers.l1(0.001)))
        model.add(BatchNormalization())
        model.add(Dense(params['n_cats'], activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=self.optim, metrics=['accuracy'])
        model.summary()
        return model

def simple_ensemble(dataset,out_path,batch_size=128,n_epochs=5):
    print(len(dataset))
    train,test=dataset.split()
    X,y,names= train.as_dataset()
    n_cats=names.n_cats()
    params={'dims':train.dim()[0],'n_cats':2}
    models=[]
    for cat_i in range(n_cats):
        model_i=SimpleNN()(params)
        y_i=names.binarize(cat_i)
        y_i=learn.to_one_hot(y_i,2)
        model_i.fit(X,y_i,epochs=n_epochs,batch_size=batch_size)
        models.append(model_i)
    files.make_dir(out_path)
    for i,model_i in enumerate(models):
        extractor_i=Model(inputs=model_i.input,
                outputs=model_i.get_layer('hidden').output)
        X,y,names=dataset.as_dataset()
        new_X=extractor_i.predict(X)
        feats_i=feats.Feats({name_j:new_X[j] 
                for j,name_j in enumerate( names)})
        out_i=f"{out_path}/{i}"
        print(out_i)
        feats_i.save(out_i)

def make_dataset(dataset,out_path,n_epochs=50):
    files.make_dir(out_path)
    dataset.save(f'{out_path}/common')
    simple_ensemble(dataset,f"{out_path}/binary",n_epochs=n_epochs)

def eff_voting(paths):
    all_results=[]
    for i,data_i in enumerate(read_multi(paths)):
        all_results.append(learn.train_model(data_i))
        print(i)
    return ens.Votes(all_results)

def read_dataset(paths):
    import gc
    common_path,binary_path=paths
    if(common_path):
        common=feats.read(common_path)[0]
    for deep_path_i in files.top_files(binary_path):
        gc.collect()
        data_i=feats.read(deep_path_i)[0]
        if(common_path):
            data_i=common+data_i
        yield data_i

def read_multi(paths):
    common_paths,binary_path=paths
    for common_i in common_paths:
        for data_j in read_dataset((common_i,binary_path)):
            yield data_j

if __name__ == "__main__":    
    paths=([None,'forest/common'],'forest/binary')
#    paths=(None,'forest/binary')
    final_votes=eff_voting(paths)
    result=final_votes.voting()
    result.report()
    print(result.get_acc())
#    files.save("forest_votes",final_votes)