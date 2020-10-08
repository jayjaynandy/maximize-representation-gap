import setup_data
import keras
import tensorflow as tf
import numpy as np
from vgg16 import VGGNet
from keras.preprocessing.image import ImageDataGenerator
import os

param = 8
num_classes = 100
batch_size  = 64
epochs = 10

model_path = './models/DPN-'
existing_path = None

if not os.path.exists('./models/'):
  os.makedirs('./models/')

print(model_path)

init_lr = 0.1
lr_drop = 20

def schedule(x):
    if x < 100:
        return 5e-4
    else:
        return 5e-6


def fn(correct, predicted):
    y_max = (tf.reduce_max(correct, axis=1) -0.5)
    y_sgm = tf.nn.sigmoid(predicted)
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted) \
            - y_max*tf.reduce_mean(y_sgm, axis=1)
            

in_dist = setup_data.Cifar100()
#data augmentation
datagen = ImageDataGenerator(rotation_range= 10, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True,
                             fill_mode="reflect")

datagen.fit(in_dist.train_data)
train_gen = datagen.flow(in_dist.train_data, in_dist.train_labels, 
                                 batch_size= batch_size)

out_dist = setup_data.Cifar10()
y_ood = out_dist.train_labels[:,1]
y_ood = keras.utils.to_categorical(y_ood, 100)
y_ood[:]= 1.0/100.0

ood_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15,
                             height_shift_range=0.15, horizontal_flip=True,
                             fill_mode="reflect")

ood_gen.fit(out_dist.train_data)
ood_gen = ood_gen.flow(out_dist.train_data, y_ood, batch_size=32)


def flow_data():
    while True:
        x0, y0 = train_gen.next()
        x1, y1 = ood_gen.next()
        xf = np.concatenate((x0, x1), axis=0)
        yf = np.concatenate((y0, y1), axis=0)        
        yield (xf, yf)

model = VGGNet()
#model.summary()

opt = keras.optimizers.Adam(lr = 5e-5)

reduce_lr = keras.callbacks.LearningRateScheduler(schedule)
model_checkpoint= keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_acc", save_best_only=True,
        save_weights_only=True, verbose=1)

if existing_path !=None:
    model.load_weights(existing_path)

model.compile(loss=fn, optimizer=opt, metrics=["accuracy"])
print("Finished compiling")

####################
# Network training #
####################
print(model.evaluate(in_dist.test_data, in_dist.test_labels))

print("Fitting the model .......... ")
his = model.fit_generator(flow_data(),
                          steps_per_epoch=int(np.ceil(in_dist.train_data.shape[0] / float(batch_size))),
                          epochs=epochs, verbose=1,
                          validation_data=(in_dist.test_data, in_dist.test_labels),
                          callbacks=[model_checkpoint, reduce_lr])

model.save_weights(model_path)