'''
C-10 Classification Task: 
    We use CIFAR-10 as in-domain data and CIFAR-100 as OOD data.

Proposed loss function: 
    goto "loss_fn" function.
    
We use the same loss function for any classification tasks.
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras import regularizers
import setup_data
import tensorflow as tf

import os
os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(9988)
np.random.seed(9988)

class cifar10vgg:
    def __init__(self):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        model.compile(loss= loss_fn, optimizer='sgd', metrics=['accuracy'])
        return model

    def train(self, model, in_dist, out_dist, model_path, existing_path = None):        
        #training parameters
        batch_size = 64
        maxepoches = 200
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        def lr_scheduler(epoch):
            lr = learning_rate * (0.5 ** (epoch // lr_drop))
            if lr < 0.0000005:
                lr = 0.0000005
            return lr
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #optimization details
        if existing_path !=None:
            model.load_weights(existing_path)
        
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss=loss_fn, optimizer=sgd, metrics=['accuracy'])

        # In-Domain training data
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        datagen.fit(in_dist.train_data)

        model_checkpoint= keras.callbacks.ModelCheckpoint(model_path, monitor="val_acc", save_best_only=True, save_weights_only=True, verbose=1)
        train_gen = datagen.flow(in_dist.train_data, in_dist.train_labels, batch_size= batch_size)

        # -------------------------- OOD training data: CIFAR 100
        y_ood = out_dist.train_labels[:,:10]
        y_ood[:] = 0.1   # uniform distribution for OOD training examples
        ood_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True)
        ood_gen.fit(out_dist.train_data)
        ood_gen = ood_gen.flow(out_dist.train_data, y_ood, batch_size=32)

        def flow_data():
            while True:
                x0, y0 = train_gen.next()
                x1, y1 = ood_gen.next()
                xf = np.concatenate((x0, x1), axis=0)
                yf = np.concatenate((y0, y1), axis=0)        
                yield (xf, yf)
        
        model.fit_generator(flow_data(),
                            steps_per_epoch=int(np.ceil(in_dist.train_data.shape[0] / float(batch_size))),
                            epochs=maxepoches,
                            validation_data=(in_dist.test_data, in_dist.test_labels), 
                            callbacks=[reduce_lr, model_checkpoint], 
                            verbose=1)

        return model
    

'''
Proposed Loss Function
'''
def loss_fn(correct, predicted):
    y_max = (tf.reduce_max(correct, axis=1) -0.5)  # Choose +0.5 for DPN+ and -0.5 for DPN-
    y_sgm = tf.nn.sigmoid(predicted)
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted) \
            - y_max*tf.reduce_mean(y_sgm, axis=1)


if __name__ == '__main__':
    cifar_in = setup_data.Cifar10()
    cifar_out = setup_data.Cifar100()
    print(cifar_in.train_data.shape, cifar_in.test_data.shape)
    model = cifar10vgg()
    model_path = './DPN_proposedLoss_l5'
    vgg = model.train(model= model.model, in_dist= cifar_in, out_dist = cifar_out,
                      model_path= model_path, existing_path= None)
    model.model.save_weights(model_path+'_final')
