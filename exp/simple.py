import sys
sys.path.append("..")
import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import convert,learn

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

def simple_ensemble(dataset,batch_size=128,n_epochs=50):
    print(len(dataset))
    model=SimpleNN()({'dims':54,'n_cats':7})
    train,test=dataset.split()
    X,y,names= train.as_dataset()
    y=learn.to_one_hot(y,7)
    model.fit(X,y,epochs=n_epochs,batch_size=batch_size)

forest=convert.forest_dataset()
simple_ensemble(forest)