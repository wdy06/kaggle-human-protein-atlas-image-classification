import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import backend as K
import tensorflow as tf

from imgaug import augmenters as iaa


class AtlasTrainDataset:
    def __init__(self, root_data_path):
        self.root_data_path = root_data_path
        self.train_dir_path = os.path.join(self.root_data_path, 'train')
        self.data_info = pd.read_csv('./data/train.csv')
        self.dataset = self.data_info_to_array(self.data_info)
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        SIZE = 299
        image = DataGenerator.load_image(self.dataset[index]['path'], (SIZE, SIZE))
        label = self.dataset[index]['labels']
        return image, label
    
    def train_test_split(self, test_size=0.2):
        train_data_info, valid_data_info = train_test_split(self.data_info, test_size=test_size, 
                                                            stratify=self.data_info['Target'].map(lambda x: x[:3] if '27' not in x else '0'), random_state=42)
        return train_data_info, valid_data_info
    
    def data_info_to_array(self, info_df):
        dataset = []
        for name, labels in zip(info_df['Id'], info_df['Target'].str.split(' ')):
            dataset.append({
                'path':os.path.join(self.train_dir_path, name),
                'labels':np.array([int(label) for label in labels])})
        return np.array(dataset)
        
class DataGenerator:

    def create_train(dataset, batch_size, shape, augument=True):
        # dataset: AtrasTrainDatasetのインスタンス
        assert shape[2] == 3
        while True:
            dataset = shuffle(dataset)
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch_images = []
                X_train_batch = dataset[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = DataGenerator.load_image(
                        X_train_batch[i]['path'], shape)
                    if augument:
                        image = DataGenerator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug
    

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
