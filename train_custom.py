import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from datetime import datetime
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import skimage.io

from PIL import Image
from scipy.misc import imread, imresize
from skimage.transform import resize
from tqdm import tqdm
import imgaug
from imgaug import augmenters as iaa

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Multiply, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import metrics
from keras.optimizers import Adam  
from keras import backend as K
import tensorflow as tf

from itertools import chain
from collections import Counter
import warnings

from model.inceptionV3 import MyInceptionV3
from utils import f1

warnings.filterwarnings("ignore")

path_to_train = './data/train/'
data = pd.read_csv('./data/train.csv')

log_dir = os.path.join('./tflog/', datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
print('create directory {}'.format(log_dir))
os.mkdir(log_dir)

# fix random seed
np.random.seed(seed=2018)
tf.set_random_seed(32)
imgaug.seed(100)

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

class DataGenerator:
    def __init__(self):
        pass
        #self.image_generator = ImageDataGenerator(rescale=1. / 255,
                                     #vertical_flip=True,
                                     #horizontal_flip=True,
                                     #rotation_range=180,
                                     #fill_mode='reflect')
    def create_train(self, dataset_info, batch_size, shape, augment=True):
        assert shape[2] == 3
        dataset_size = len(dataset_info)
        while True:
            #random_indexes = np.random.choice(len(dataset_info), batch_size)
            permutation = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                random_indexes = permutation[start : start+batch_size]
                batch_images1 = np.empty((batch_size, shape[0], shape[1], shape[2]))
                batch_labels = np.zeros((batch_size, 28))
                for i, idx in enumerate(random_indexes):
                    image1 = self.load_4ch_image(
                        dataset_info[idx]['path'], shape)
                    if augment:
                        image1 = self.augment(image1)
                    batch_images1[i] = image1/255.
                    batch_labels[i][dataset_info[idx]['labels']] = 1
                yield [batch_images1], batch_labels
            
    
    def load_image(self, path, shape):
        image_red_ch = skimage.io.imread(path+'_red.png')
        image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        image_green_ch = skimage.io.imread(path+'_green.png')
        image_blue_ch = skimage.io.imread(path+'_blue.png')

        image1 = np.stack((
            image_green_ch, 
            image_red_ch, 
            image_blue_ch), -1)
        image1 = resize(image1, (shape[0], shape[1], 3), mode='reflect')
        return image1.astype(np.float)

    def load_4ch_image(self, path, shape):
        use_channel = [0, 1, 2]
        image = np.array(Image.open(path+'_rgby.png'))
        image = image[:,:,use_channel]
        image = resize(image, (shape[0], shape[1], 3), mode='reflect')
        return image

    def augment(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

# create train datagen
train_datagen = DataGenerator()

generator = train_datagen.create_train(
    train_dataset_info, 5, (299,299,3))

# from https://www.kaggle.com/kmader/rgb-transfer-learning-with-inceptionv3-for-protein
data['target_list'] = data['Target'].map(lambda x: [int(a) for a in x.split(' ')])
all_labels = list(chain.from_iterable(data['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
data['target_vec'] = data['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(data, 
                 test_size = 0.2, 
                  # hack to make stratification work                  
                 stratify = data['Target'].map(lambda x: x[:3] if '27' not in x else '0'), random_state=42)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')
train_df.to_csv('train_part.csv')
valid_df.to_csv('valid_part.csv')

train_dataset_info = []
for name, labels in zip(train_df['Id'], train_df['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
valid_dataset_info = []
for name, labels in zip(valid_df['Id'], valid_df['Target'].str.split(' ')):
    valid_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
valid_dataset_info = np.array(valid_dataset_info)
print(train_dataset_info.shape, valid_dataset_info.shape)


keras.backend.clear_session()

tfconfig = tfconfig = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
sess = tf.Session(config=tfconfig)
K.set_session(sess)

model = MyInceptionV3.create_model(
    input_shape=(299,299,3), 
    n_out=28)

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['acc', f1])

model.summary()


epochs = 100; batch_size = 16
checkpointer = ModelCheckpoint(
    os.path.join(log_dir,'Xception.model'), 
    monitor='val_f1',
    mode='max',
    verbose=2, 
    save_best_only=True,
    save_weights_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.3, patience=5,
                                   verbose=1, mode='max', epsilon=0.0001)
early = EarlyStopping(monitor="val_f1",
                      mode="max",
                      patience=12)

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

# create train and valid datagens
train_generator = train_datagen.create_train(
    train_dataset_info, batch_size, (299,299,3))
validation_generator = train_datagen.create_train(
    valid_dataset_info, batch_size, (299,299,3), augment=False)
#K.set_value(model.optimizer.lr, 0.0002)
# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_df)//batch_size,
    validation_data=validation_generator,
    validation_steps=len(valid_df)//batch_size//10,
    epochs=epochs, 
    verbose=1,
    callbacks=[checkpointer, reduce_lr, early, tensorboard])

submit = pd.read_csv('./data/sample_submission.csv')

# load best model
model.load_weights(os.path.join(log_dir, 'Xception.model'))
predicted = []
from tqdm import tqdm_notebook
for name in tqdm(submit['Id']):
    path = os.path.join('./data/test/', name)
    image1 = train_datagen.load_4ch_image(path, (299,299,3))/255.
    score_predict = model.predict([image1[np.newaxis]])[0]
    label_predict = np.arange(28)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('submission_one_branches_xception.csv', index=False)


