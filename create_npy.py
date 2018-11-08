# fix random seed
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(32)

import argparse
import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from tqdm import tqdm_notebook


from itertools import chain
from collections import Counter
import warnings

from model.inceptionV3 import MyInceptionV3
from utils import f1


parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

path_to_train = './data/train/'
data = pd.read_csv('./data/train.csv')


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

def load_4ch_image(path, shape):
    use_channel = [0, 1, 2]
    image = np.array(Image.open(path+'_rgby.png'))
    image = image[:,:,use_channel]
    image = resize(image, (shape[0], shape[1], 3), mode='reflect')
    return image


# load data on memory
input_shape = (512, 512, 3)

n_out = 28

debug_size = 100
if args.debug:
    train_size = debug_size
    valid_size = debug_size
else:
    train_size = len(train_df)
    valid_size = len(valid_df)

x_train = np.empty((train_size, input_shape[0], input_shape[1], input_shape[2]))
y_train = np.zeros((train_size, n_out))

print('loading train data ...')
for idx, (name, label) in tqdm(enumerate(zip(train_df['Id'], train_df['target_vec']))):
    if args.debug and idx >= debug_size:
        break
    #print(idx, name, label)
    path = os.path.join('./data/train/', name)
    x_train[idx] = load_4ch_image(path, input_shape)/255.
    y_train[idx][label] = 1

x_valid = np.empty((valid_size, input_shape[0], input_shape[1], input_shape[2]))
y_valid = np.zeros((valid_size, n_out))
print('loading validation data ...')
for idx, (name, label) in tqdm(enumerate(zip(valid_df['Id'], valid_df['target_vec']))):
    if args.debug and idx >= debug_size:
        break
    #print(idx, name, label)
    path = os.path.join('./data/train/', name)
    x_valid[idx] = load_4ch_image(path, input_shape)/255.
    y_valid[idx][label] = 1

# save loaded data to npy
suffix_str = 'rgb_512'

if args.debug:
    suffix_str = 'debug_' + suffix_str
np.save('./data/npy_data/x_train_{}.npy'.format(suffix_str), x_train)
np.save('./data/npy_data/y_train_{}.npy'.format(suffix_str), y_train)
np.save('./data/npy_data/x_valid_{}.npy'.format(suffix_str), x_valid)
np.save('./data/npy_data/y_valid_{}.npy'.format(suffix_str), y_valid)
print('saved npy file')

