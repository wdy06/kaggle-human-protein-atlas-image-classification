import os
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm


data_info = pd.read_csv('./data/train.csv')
train_dir = './data/train'

for name in tqdm(data_info['Id']):
    path = os.path.join(train_dir, name)
    image_red_ch = Image.open(path+'_red.png')
    image_yellow_ch = Image.open(path+'_yellow.png')
    image_green_ch = Image.open(path+'_green.png')
    image_blue_ch = Image.open(path+'_blue.png')
    image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch),
        np.array(image_yellow_ch)), -1)

    save_path = '{}_rgby.png'.format(path)
    print('save to {}'.format(save_path))
    Image.fromarray(image).save(save_path)




