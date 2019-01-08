import concurrent.futures
import os
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm
import cv2

def convert_jpg_to_png(name, dir_path, size):
    if name in [4270, 15573, 37343]:
        return
    path = os.path.join(dir_path, str(name))
    colors = ['red','green','blue','yellow']
    #flags = cv2.IMREAD_GRAYSCALE
    for color in colors:
        jpg_path = path+'_'+color+'.jpg'
        print(jpg_path)
        image = cv2.imread(jpg_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if color == 'red':
            image = image[:,:,0]
        elif color == 'green':
            image = image[:,:,1]
        elif color == 'blue':
            image = image[:,:,2]
        elif color == 'yellow':
            image = image[:,:,0]
        else:
            raise ValueError('unknown color')
        #image = cv2.imread(jpg_path, flags).astype(np.float32)/255
        #image = Image.open(jpg_path)
        image = cv2.resize(image, (size, size),cv2.INTER_AREA)
        
        save_path = '{}_{}.png'.format(path, color)
        #print('save to {}'.format(save_path))
        image = Image.fromarray(image)
        #image = image.convert("L")
        image.save(save_path)

        
def main():
    size = 512
    data_info = pd.read_csv('./data/augment.csv', names=['Id', 'Target'])
    test_dir = './data/aug_images'
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for name in tqdm(data_info['Id']):
        #     if name < 37342:
        #         continue
            executor.submit(convert_jpg_to_png, name, test_dir, size)

    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    