# fix random seed
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(32)

import argparse
import os, sys
from datetime import datetime
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import skimage.io
import shutil

from PIL import Image
from scipy.misc import imread, imresize
from skimage.transform import resize
from tqdm import tqdm
import imgaug
from imgaug import augmenters as iaa
from tqdm import tqdm_notebook

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
from model_multi_gpu import ModelMGPU

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument('--gpu', '-g', default='1', type=str,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', type=str, default='xception',
                    help='cnn model')
parser.add_argument('--lr', '-l', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--optimizer', '-o', type=str, default='adam',
                    help='optimizer')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--multi", type=int, default=1, help="train using multi GPUs")
args = parser.parse_args()

if not args.multi:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# set tf session
keras.backend.clear_session()

tfconfig = tfconfig = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
sess = tf.Session(config=tfconfig)
K.set_session(sess)

path_to_train = './data/train/'

# fix random seed
imgaug.seed(100)



def load_4ch_image(path, shape):
    use_channel = [0, 1, 2]
    image = np.array(Image.open(path+'_rgby.png'))
    image = image[:,:,use_channel]
    image = resize(image, (shape[0], shape[1], 3), mode='reflect')
    return image


# load data on memory
input_shape = (299, 299, 3)
n_out = 28

if args.debug:
    print('loading train debug data ...')
    x_train = np.load('./data/x_train_debug.npy')
    y_train = np.load('./data/y_train_debug.npy')
    print('loading validation debug data ...')
    x_valid = np.load('./data/x_valid_debug.npy')
    y_valid = np.load('./data/y_valid_debug.npy')
else:
    print('loading train data ...')
    x_train = np.load('./data/x_train.npy')
    y_train = np.load('./data/y_train.npy')
    print('loading validation data ...')
    x_valid = np.load('./data/x_valid.npy')
    y_valid = np.load('./data/y_valid.npy')

# create model
if args.model == 'xception':
    from model.xception import MyXception
    single_model = MyXception.create_model(
        input_shape=input_shape, 
        n_out=n_out)
elif args.model == 'inceptionV3':
    from model.inceptionV3 import MyInceptionV3
    single_model = MyInceptionV3.create_model(
        input_shape=input_shape, 
        n_out=n_out)
elif args.model == 'resnet50':
    from model.resnet50 import MyResNet50
    single_model = MyResNet50.create_model(
        input_shape=input_shape, 
        n_out=n_out)
else:
    raise ValueError('model name is invalid')


if args.multi > 1:
    print('using multi GPUs')
    model = ModelMGPU(single_model, args.multi)
else:
    print('using single GPU')
    model = single_model



model.compile(
    loss='binary_crossentropy', 
    optimizer=args.optimizer,
    metrics=['acc', f1])

K.set_value(model.optimizer.lr, args.lr)
model.summary()


if args.debug:
    epochs = 1
else:
    epochs = 100
batch_size = 32 * args.multi

# make log directory
log_dir_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')+'-{}-{}-lr{}-B{}'.format(args.model, args.optimizer, args.lr, batch_size)
if args.debug:
    log_dir_name = 'debug-' + log_dir_name
log_dir = os.path.join('./tflog/', log_dir_name)
print('create directory {}'.format(log_dir))
os.mkdir(log_dir)




checkpointer = ModelCheckpoint(
    os.path.join(log_dir,'{}.model'.format(model.name)), 
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

# create train and valid datagen
datagen = ImageDataGenerator(
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)

# train model
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                     epochs=epochs,
                     validation_data=(x_valid, y_valid),
                     workers=20,
                     verbose=1,
                     callbacks=[checkpointer, reduce_lr, early, tensorboard])


submit = pd.read_csv('./data/sample_submission.csv')

# load best model
single_model.load_weights(os.path.join(log_dir, '{}.model'.format(model.name)))
predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('./data/test/', name)
    image1 = load_4ch_image(path, (299,299,3))/255.
    score_predict = single_model.predict([image1[np.newaxis]])[0]
    label_predict = np.arange(28)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
save_file = 'submission_{}.csv'.format(model.name)
submit.to_csv(save_file, index=False)
print('saved to {}'.format(save_file))

if args.debug:
    print('remove {}'.format(log_dir))
    shutil.rmtree(log_dir)


