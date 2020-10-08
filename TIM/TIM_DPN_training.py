'''
Please provide the appropriate location for traini_dir, ood_dir and 
validation_dir.

'''
import keras
import tensorflow as tf
import numpy as np
from vgg16 import VGGNet
from keras.preprocessing.image import ImageDataGenerator
import os

num_classes = 200
batch_size  = 86
test_batch_size=500
epochs = 5

model_path = './models/DPN-'
existing_path = model_path

if not os.path.exists('./models/'):
  os.makedirs('./models/')
  
train_dir = "./data/tiny-imagenet-200/train/"
ood_dir = "./imagenet/ood_data/"
validation_dir = "./data/tiny-imagenet-200/val/"
num_train_img = 100000
num_test_img =  10000


learning_rate = 0.1

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

#data augmentation
datagen = ImageDataGenerator(rescale = 1./255, rotation_range=10,
                             width_shift_range=0.10, height_shift_range=0.10,
                             horizontal_flip=True, fill_mode="reflect")

train_generator = datagen.flow_from_directory(
            directory = train_dir,
            target_size=(64, 64),            
            batch_size = batch_size,
            class_mode = 'categorical')


#data augmentation
ood_gen = ImageDataGenerator(rescale=1./255, rotation_range=15,
                             width_shift_range=0.15, height_shift_range=0.15,
                             horizontal_flip=True, fill_mode="reflect")

ood_gen = ood_gen.flow_from_directory(
            directory = ood_dir,
            target_size=(64, 64),            
            batch_size = 43,
            class_mode = 'categorical')


def flow_data():
    while True:
        x0, y0 = train_generator.next()
        x1, y1 = ood_gen.next()
        y1 = keras.utils.to_categorical(y1[:,0], 200)
        y1[:]= 1.0/200.0
        xf = np.concatenate((x0, x1), axis=0)
        yf = np.concatenate((y0, y1), axis=0)        
        yield (xf, yf)


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size = 500,
        class_mode = 'categorical',
        shuffle = False)


model = VGGNet()
#model.summary()

opt = keras.optimizers.Adam(lr = 5e-4)

reduce_lr = keras.callbacks.LearningRateScheduler(schedule)
model_checkpoint= keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_acc", save_best_only= True,
        save_weights_only=True, verbose=1)

if existing_path !=None:
    model.load_weights(existing_path)
        
model.compile(loss=fn, optimizer=opt, metrics=["accuracy"])
print("Finished compiling")

####################
# Network training #
####################

print("Fitting the model .......... ")
his = model.fit_generator(flow_data(),
                          steps_per_epoch=int(np.ceil(num_train_img / float(batch_size))),
                          epochs=epochs, 
                          verbose=1,
                          validation_data= validation_generator,
                          validation_steps= int(np.ceil(num_test_img / float(test_batch_size))),
                          callbacks=[model_checkpoint, reduce_lr])

