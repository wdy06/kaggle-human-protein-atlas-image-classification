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
from sklearn.metrics import f1_score
import shutil

import threading
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
from model_multi_gpu import ModelMGPU
from focal_loss import focal_loss
from util_threshold import find_thresh
from keras_tta import TTA_ModelWrapper
from clr_callback import CyclicLR

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument('--gpu', '-g', default='1', type=str,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', type=str, default='xception',
                    help='cnn model')
parser.add_argument('--lr', '-l', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--loss', type=str, default='binary',
                    help='loss function')
parser.add_argument('--batch', '-B', type=int, default=32,
                    help='batch size')
parser.add_argument('--size', '-s', type=int, default=256,
                    help='image size')
parser.add_argument('--optimizer', '-o', type=str, default='adam',
                    help='optimizer')
parser.add_argument('--weight', '-w', type=str, default=None,
                    help='pretrained model weight')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--use_clr", help="run using cycling leaning rate",
                    action="store_true")
parser.add_argument("--multi", type=int, default=1, help="train using multi GPUs")
args = parser.parse_args()

if args.multi == 1:
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
data_csv = pd.read_csv('./data/train.csv')

data_csv['target_list'] = data_csv['Target'].map(lambda x: [int(a) for a in x.split(' ')])
all_labels = list(chain.from_iterable(data_csv['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
data_csv['target_vec'] = data_csv['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(data_csv,
                     test_size = 0.2,
                    # hack to make stratification work
                    stratify = data_csv['Target'].map(lambda x: x[:3] if '27' not in x else '0'), random_state=42)
print(train_df.shape[0], 'training data')
print(valid_df.shape[0], 'validation data')
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

def load_4ch_image(path, shape):
    use_channel = [0, 1, 2]
    image = np.array(Image.open(path+'_rgby.png'))
    image = image[:,:,use_channel]
    image = resize(image, (shape[0], shape[1], 3), mode='reflect')
    return image

class threadsafe_iter:
    
    """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.abs    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class DataGenerator:
    def __init__(self, datagen, stats):
        self.datagen = datagen
        self.stats = stats

    @threadsafe_generator
    def create_train(self, dataset_info, batch_size, shape, augment=False):
        assert len(shape) == 3
        assert shape[2] == 3
        dataset_size = len(dataset_info)
        while True:
            permutation = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                random_indexes = permutation[start : start+batch_size]
                batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
                batch_labels = np.zeros((batch_size, 28))
                for i, idx in enumerate(random_indexes):
                    image = load_4ch_image(dataset_info[idx]['path'], shape)
                    batch_images[i] = image
                    # normalize
                    batch_images = normalize(batch_images, self.stats)
                    batch_labels[i][dataset_info[idx]['labels']] = 1
                if augment:
                    # augmentation
                    batch_images, batch_labels = next(self.datagen.flow(batch_images, y=batch_labels, batch_size=batch_size))
                yield batch_images, batch_labels

# load data on memory
input_shape = (args.size, args.size, 3)
n_out = 28


# mean and std of data
if args.size == 299:
    train_stats = np.array([[0.08044203, 0.05263003, 0.05474688],
                            [0.12098549, 0.07966491, 0.13656638]])
    test_stats = np.array([[0.05908037, 0.04532997, 0.04065239],
                           [0.09605538, 0.07202642, 0.10485397]])
elif args.size == 512:
    train_stats = np.array([[0.0804419,  0.05262986, 0.05474701],
                            [0.13000701, 0.08796628, 0.1386317]])
    test_stats = np.array([[0.05908022, 0.04532852, 0.04065233],
                           [0.10371015, 0.07984633, 0.10664798]])
else:
    raise ValueError('stats data needed.')

    
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
elif args.model == 'wideResnet':
    from model.wide_residual_network import create_model
    single_model = create_model(input_shape=input_shape, n_out=n_out, dropout=0.5)
else:
    raise ValueError('model name is invalid')

if args.weight is not None:
    print('run transfer learning')
    old_input_shape = (299, 299, 3)
    from model.xception import MyXception
    old_model = MyXception.create_model(
        input_shape=old_input_shape, 
        n_out=n_out)
    old_model.load_weights(args.weight)

    for layer, old_layer in zip(single_model.layers[1:], old_model.layers[1:]):
        layer.set_weights(old_layer.get_weights())

if args.multi > 1:
    print('using multi GPUs')
    model = ModelMGPU(single_model, args.multi)
else:
    print('using single GPU')
    model = single_model

if args.loss == 'binary':
    loss_func = 'binary_crossentropy'
elif args.loss == 'focal':
    loss_func = focal_loss(alpha=.25, gamma=2)
else:
    raise ValueError('invalid loss function')


batch_size = args.batch * args.multi

use_clr = args.use_clr
if use_clr:
    print('use cycling learning rate')
    clr = CyclicLR(base_lr=args.lr/10, max_lr=args.lr,
                            step_size=np.round(len(x_train)/batch_size * 5), mode='triangular2')

model.compile(
    loss=loss_func, 
    optimizer=args.optimizer,
    metrics=['acc', f1])

K.set_value(model.optimizer.lr, args.lr)
model.summary()


if args.debug:
    epochs = 1
else:
    epochs = 100

# set augment times when test time augmentation
if args.debug:
    aug_times = 2
else:
    aug_times = 8

# make log directory
log_dir_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')+'-{}-{}-lr{}-B{}-s{}-{}-brute-tta{}-znorm'.format(
				args.model, 
				args.optimizer, 
				args.lr, 
				batch_size,
				args.size,
                                args.loss,
                                aug_times)
if args.debug:
    log_dir_name = 'debug-' + log_dir_name
if use_clr:
    log_dir_name = log_dir_name + '-clr'
if args.weight is not None:
    log_dir_name = log_dir_name + '-transfer'
log_dir = os.path.join('./tflog/', log_dir_name)
print('create directory {}'.format(log_dir))
os.mkdir(log_dir)




checkpointer = ModelCheckpoint(
    os.path.join(log_dir,'{}.model'.format(args.model)), 
    monitor='val_loss',
    mode='min',
    verbose=2, 
    save_best_only=True,
    save_weights_only=True)

if use_clr:
    factor=1.
else:
    factor=0.3
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5,
                                   verbose=1, mode='min', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=12)

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

# create train and valid datagen
imagegen = ImageDataGenerator(
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    # set mode for filling points outside the input boundaries
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=20,
    zoom_range=[0.8, 1.2],
    fill_mode='reflect',
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)

data_gen = DataGenerator(imagegen, train_stats)
train_generator = data_gen.create_train(train_dataset_info, batch_size, input_shape, augment=True)
valid_generator = data_gen.create_train(valid_dataset_info, batch_size, input_shape, augment=False)
# train model

#defile callback list
callback_list = [checkpointer, reduce_lr, early, tensorboard]
if use_clr:
    callback_list.append(clr)

history = model.fit_generator(train_generator,
                     epochs=epochs,
                     steps_per_epoch=len(train_dataset_info)//batch_size,
                     validation_data=valid_generator,
                     validation_steps=len(valid_dataset_info)//batch_size//10,
                     workers=20,
                     verbose=1,
                     callbacks=callback_list)

# load best model
single_model.load_weights(os.path.join(log_dir, '{}.model'.format(args.model)))

# use test time augmentation
print('use test time augmentation')
single_model = TTA_ModelWrapper(single_model)

# decide best threshold
f1_list = []
threshold_list = list(np.arange(0, 1, 0.01))
pred_matrix = single_model.predict_tta(x_valid, aug_times=aug_times)
for th in threshold_list:
    pred_label_matrix = np.zeros(pred_matrix.shape)
    pred_label_matrix[pred_matrix >= th] = 1
    f1 = f1_score(y_valid, pred_label_matrix, average='macro')
    print('threshold: {}, f1_score: {}'.format(th, f1))
    f1_list.append(f1)

max_f1 = max(f1_list)
best_th = threshold_list[np.argmax(np.array(f1_list))]
print('best threshold: {}, best f1 score: {}'.format(best_th, max_f1))

# find best threshold2
ths = find_thresh(pred_matrix, y_valid)
print(ths)

#pred_matrix = single_model.predict_tta(x_valid)
pred_label_matrix = np.zeros(pred_matrix.shape)
pred_label_matrix[pred_matrix >= ths] = 1
f1 = f1_score(y_valid, pred_label_matrix, average='macro')
print('using brute force best  threshold, f1_score: {}'.format(f1))

submit = pd.read_csv('./data/sample_submission.csv')

predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('./data/test/', name)
    image = load_4ch_image(path, (input_shape[0], input_shape[1] ,input_shape[2]))
    image = image[np.newaxis]
    image = normalize(image, test_stats)
    score_predict = single_model.predict_tta(image, aug_times=aug_times)[0]
    #score_predict = single_model.predict_tta([image], aug_times=aug_times)[0]
    label_predict = np.arange(28)[score_predict>=best_th]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
save_file = 'submission_{}_th{}_valf1_{}.csv'.format(model.name, best_th, max_f1)
save_file_path = os.path.join(log_dir, save_file)
submit.to_csv(save_file_path, index=False)
print('saved to {}'.format(save_file_path))


