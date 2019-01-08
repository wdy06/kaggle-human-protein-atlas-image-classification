import concurrent.futures
from itertools import repeat
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
import sys
sys.path.append("/tmp/fastai/old")

#from utils import load_4ch_image
from utils_pytorch import open_rgby

def calculate_stats(name, dir_path):
    #print(name)
    image = open_rgby(dir_path, name)
    image_mean = np.mean(image.reshape(-1, 4), axis=0)
    image_std = np.std(image.reshape(-1, 4), axis=0)
    
    return image_mean, image_std

def main():
    #data_info = pd.read_csv('./data/train.csv')
    data_info = pd.read_csv('./data/augment_train.csv')
    #train_dir = './data/train'
    train_dir = './data/all_train'
    test_dir = './data/test'

    mean_sum = 0
    std_sum = 0
    print('caluculating train stats...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for image_mean, image_std in executor.map(calculate_stats, data_info['Id'], repeat(train_dir)):
        #     path = os.path.join(train_dir, name)
        #     image = load_4ch_image(path, size)
            mean_sum += image_mean
            std_sum += image_std

        print('train stats')
        print(mean_sum/len(data_info['Id']))
        print(std_sum/len(data_info['Id']))

    test_mean_sum = 0
    test_std_sum = 0
    submit = pd.read_csv('./data/sample_submission.csv')
    print('caluculating test stats...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for image_mean, image_std in executor.map(calculate_stats, submit['Id'], repeat(test_dir)):
        #     path = os.path.join(test_dir, name)
        #     image = load_4ch_image(path, size)
            test_mean_sum += image_mean
            test_std_sum += image_std

    print('test stats')
    print(test_mean_sum/len(submit['Id']))
    print(test_std_sum/len(submit['Id']))

if __name__ == '__main__':
    main()
