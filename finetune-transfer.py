import os
import sys
import glob
import argparse

from keras import __version__
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten
from myimage import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from dogcat_data import generators, get_nb_files

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
config.img_width = 150
config.img_height = 150
config.epochs = 50
config.batch_size = 32



train_dir = "../dogcat-data/train"
val_dir = "../dogcat-data/validation"

nb_train_samples = get_nb_files(train_dir)
nb_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(val_dir)

# data prep
train_generator, validation_generator = generators(
    preprocess_input, config.img_width, config.img_height, config.batch_size)

# setup model
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))


model = Sequential()
model.add(conv_base)
model.add(Flatten(input_shape=conv_base.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

# transfer learning
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-5),
              metrics=['acc'])

train_generator, validation_generator = generators(
    preprocess_input, config.img_width, config.img_height, config.batch_size)


model.fit_generator(
    train_generator,
    epochs=config.epochs,
    steps_per_epoch=nb_train_samples // config.batch_size,
    validation_data=validation_generator,
    validation_steps=nb_val_samples // config.batch_size,
    callbacks=[WandbCallback()])

model.save('transfered_finetune.h5')
