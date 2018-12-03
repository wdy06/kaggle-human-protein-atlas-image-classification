import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from utils import load_4ch_image

data_info = pd.read_csv('./data/train.csv')
train_dir = './data/train'

size = 299

print('size: {}'.format(size))

mean_sum = 0
std_sum = 0
for name in tqdm(data_info['Id']):
    path = os.path.join(train_dir, name)
    image = load_4ch_image(path, size)
    mean_sum += np.mean(image.reshape(-1, 4), axis=0)
    std_sum += np.std(image.reshape(-1, 4), axis=0)
    
print('train stats')
print(mean_sum/len(data_info['Id']))
print(std_sum/len(data_info['Id']))

test_mean_sum = 0
test_std_sum = 0
submit = pd.read_csv('./data/sample_submission.csv')
for name in tqdm(submit['Id']):
    path = os.path.join('./data/test/', name)
    image = load_4ch_image(path, size)
    test_mean_sum += np.mean(image.reshape(-1, 4), axis=0)
    test_std_sum += np.std(image.reshape(-1, 4), axis=0)

print('test stats')
print(test_mean_sum/len(submit['Id']))
print(test_std_sum/len(submit['Id']))
