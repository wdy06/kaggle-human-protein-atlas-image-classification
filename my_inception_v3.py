import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import numpy as np
import pandas as pd
from datetime import datetime
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle

import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

import utils

log_dir = os.path.join('./tflog/', datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
print('create directory {}'.format(log_dir))
os.mkdir(log_dir)

SIZE = 299

# Load dataset info
train_dataset = utils.AtlasTrainDataset('./data/')

tfconfig = tfconfig = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
sess = tf.Session(config=tfconfig)
K.set_session(sess)
    
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

# create callbacks list
epochs = 20; batch_size = 16
checkpoint = ModelCheckpoint('./working/InceptionV3.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
                                   verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6)
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [checkpoint, early, reduceLROnPlat, tensorboard]

# split data into train, valid

train_info, valid_info = train_dataset.train_test_split()
train_data = train_dataset.data_info_to_array(train_info)
valid_data = train_dataset.data_info_to_array(valid_info)

# create train and valid datagens
train_generator = utils.DataGenerator.create_train(
    train_data, batch_size, (SIZE,SIZE,3), augument=True)
validation_generator = utils.DataGenerator.create_train(
    valid_data, 32, (SIZE,SIZE,3), augument=False)

# warm up model

model = create_model(
    input_shape=(SIZE,SIZE,3), 
    n_out=28)
'''
for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(1e-03),
    metrics=['acc', utils.f1])
# model.summary()
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_data)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_data)) / float(batch_size)),
    epochs=2, 
    verbose=1,
    max_queue_size=40,
    use_multiprocessing=True,
    workers=20)
'''

# train all layers
'''
for layer in model.layers:
    layer.trainable = True
'''
model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=1e-3),
            metrics=['acc', utils.f1])
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_data)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_data)) / float(batch_size)),
    epochs=epochs, 
    verbose=1,
    max_queue_size=40,
    use_multiprocessing=True,
    workers=20,
    callbacks=callbacks_list)
    
# Create submit
submit = pd.read_csv('./data/sample_submission.csv')
predicted = []
draw_predict = []
model.load_weights('./working/InceptionV3.h5')
for name in tqdm(submit['Id']):
    path = os.path.join('./data/test/', name)
    image = utils.DataGenerator.load_image(path, (SIZE,SIZE,3))/255.
    score_predict = model.predict(image[np.newaxis])[0]
    draw_predict.append(score_predict)
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv('submit_my_InceptionV3.csv', index=False)
