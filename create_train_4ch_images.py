import concurrent.futures
import os
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

def generate_4ch_image(name, dir_path):
    if name in [4270, 15573, 37343]:
        return
    path = os.path.join(dir_path, str(name))
    #print(path)
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

def main():
    csv_path = './data/augment.csv'
    data_info = pd.read_csv(csv_path, names=['Id', 'Target'])

    train_dir = './data/aug_images'
    #train_dir = './data/train_full_size'

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for name in tqdm(data_info['Id']):
            executor.submit(generate_4ch_image, name, train_dir)
        


if __name__ == '__main__':
    main()


