# fix random seed
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(32)

import argparse
import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import copy
from datetime import datetime
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import skimage.io
from sklearn.metrics import f1_score
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
from utils import f1, normalize
from util_threshold import find_thresh
from model_multi_gpu import ModelMGPU
from keras_tta import TTA_ModelWrapper

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument('--model', '-m', type=str, help='cnn model')
parser.add_argument('--weight', '-w', type=str, help='path to model weight')
args = parser.parse_args()

# set tf session
keras.backend.clear_session()

tfconfig = tfconfig = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
sess = tf.Session(config=tfconfig)
K.set_session(sess)

def load_4ch_image(path, shape):
    use_channel = [0, 1, 2]
    image = np.array(Image.open(path+'_rgby.png'))
    image = image[:,:,use_channel]
    image = resize(image, (shape[0], shape[1], shape[2]), mode='reflect')
    return image


# load data on memory
input_shape = (299, 299, 3)
#input_shape = (256, 256, 3)
n_out = 28

print('loading validation data ...')
x_valid = np.load('./data/npy_data/x_valid_rgb_{}.npy'.format(input_shape[0]))
y_valid = np.load('./data/npy_data/y_valid_rgb_{}.npy'.format(input_shape[0]))

# mean and std of data
train_stats = np.array([[0.08044203, 0.05263003, 0.05474688],
                        [0.12098549, 0.07966491, 0.13656638]])
test_stats = np.array([[0.05908037, 0.04532997, 0.04065239],
                       [0.09605538, 0.07202642, 0.10485397]])

x_valid = normalize(x_valid, train_stats)

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


model = single_model


submit = pd.read_csv('./data/sample_submission.csv')

# load model weight
single_model.load_weights(args.weight)

# use test time augmentation
single_model = TTA_ModelWrapper(single_model)

aug_times=16
# decide best threshold
f1_list = []
threshold_list = list(np.arange(0, 1, 0.05))
print('predicting with tta ...')
pred_matrix = single_model.predict_tta(x_valid, aug_times=aug_times)
for th in threshold_list:
    pred_label_matrix = np.zeros(pred_matrix.shape)
    pred_label_matrix[pred_matrix >= th] = 1
    f1 = f1_score(y_valid, pred_label_matrix, average='macro')
    print('threshold: {}, f1_score: {}'.format(th, f1))
    f1_list.append(f1)

max_f1 = max(f1_list)
best_th = threshold_list[np.argmax(np.array(f1_list))]
print('best const threshold: {}, best f1 score: {}'.format(best_th, max_f1))

# find best threshold2
ths = find_thresh(pred_matrix, y_valid)
print(ths)

#pred_matrix = single_model.predict_tta(x_valid)
pred_label_matrix = np.zeros(pred_matrix.shape)
pred_label_matrix[pred_matrix >= ths] = 1
f1 = f1_score(y_valid, pred_label_matrix, average='macro')
print('using brute force best  threshold, f1_score: {}'.format(f1))

predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('./data/test/', name)
    image = load_4ch_image(path, (input_shape[0], input_shape[1] ,input_shape[2]))
    image = image[np.newaxis]
    image = normalize(image, test_stats)
    score_predict = single_model.predict_tta(image, aug_times=aug_times)[0]
    label_predict = np.arange(28)[score_predict>=ths]
    #label_predict = np.arange(28)[score_predict>=best_th]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
save_file = 'submission_maked.csv'.format(model.name)
submit.to_csv(save_file, index=False)
print('saved to {}'.format(save_file))



