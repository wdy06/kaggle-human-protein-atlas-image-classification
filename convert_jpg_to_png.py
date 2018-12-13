import os
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm
import cv2


data_info = pd.read_csv('./data/augment.csv', names=['Id', 'Target'])

test_dir = './data/aug_images'

for name in tqdm(data_info['Id']):
#     if name < 37342:
#         continue
    if name in [4270, 15573, 37343]:
        continue
    path = os.path.join(test_dir, str(name))
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    for color in colors:
        jpg_path = path+'_'+color+'.jpg'
        print(jpg_path)
        image = cv2.imread(jpg_path, flags)
        #image = cv2.imread(jpg_path, flags).astype(np.float32)/255
        #image = Image.open(jpg_path)
        
        save_path = '{}_{}.png'.format(path, color)
        #print('save to {}'.format(save_path))
        image = Image.fromarray(image)
        image = image.convert("L")
        image.save(save_path)
