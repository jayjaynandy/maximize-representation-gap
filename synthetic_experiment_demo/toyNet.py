from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Input

def ToyNet(nb_classes=3, img_dim= (2,)):
    model_input = Input(shape=img_dim)    
    x = Dense(125, activation='relu')(model_input)
    x = Dense(125, activation='relu')(x)
    x = Dense(nb_classes)(x)    
    toyNet = Model(inputs=[model_input], outputs=[x], name="DenseNet")
    return toyNet
