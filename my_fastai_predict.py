import argparse
import sys
print(sys.path)
import seaborn as sns
sns.set()

sys.path.append("/tmp/fastai/old")

from fastai.conv_learner import *
from fastai.dataset import *

from datetime import datetime
import pandas as pd
import numpy as np
np.random.seed(seed=32)
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

#import tensorflow as tf
#from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from tensorboard_cb import TensorboardLogger
import torch
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
import utils_pytorch
import custom_model


parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

nw = 20   #number of workers for data loader
#arch = resnet34 #specify target architecture
arch = custom_model.get_model('se_resnext50')

train_names = list({f[:36] for f in os.listdir(utils_pytorch.TRAIN)})
test_names = list({f[:36] for f in os.listdir(utils_pytorch.TEST)})
# tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)

data_info = pd.read_csv(utils_pytorch.LABELS)
tr_n, val_n = train_test_split(data_info, test_size = 0.1, 
                 stratify = data_info['Target'].map(lambda x: x[:3] if '27' not in x else '0'), random_state=42)
tr_n = tr_n['Id'].tolist()
val_n = val_n['Id'].tolist()


def get_data(sz,bs,stats=None):
    #data augmentation
    aug_tfms = [RandomRotate(45, tfm_y=TfmType.NO),
                RandomFlip(),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    #mean and std in of each channel in the train set
    #stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
    if stats is None:
        #stats = A([0.0804419, 0.05262986, 0.05474701, 0.08270896], [0.13000701, 0.08796628, 0.1386317, 0.12718021]) # calulate myself
        #stats = A([0.06734, 0.05087, 0.03266, 0.09257],[0.11997, 0.10335, 0.10124, 0.1574 ]) # include external data, caluculated by moriyama
        stats = A([0.1057, 0.06651, 0.06325, 0.09928],[0.15266, 0.10139, 0.15967, 0.14573]) # include external data, caluculated by myself
    else:
        stats = A(stats)
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(utils_pytorch.pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],utils_pytorch.TRAIN), 
                (val_n,utils_pytorch.TRAIN), tfms, test=(test_names,utils_pytorch.TEST))
    md = ImageData(utils_pytorch.PATH, ds, bs, num_workers=nw, classes=None)
    return md

sz = 512 #image size
bs = 32  #batch size


md = get_data(sz,bs)
#learner = utils_pytorch.ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learner.models.model = torch.nn.DataParallel(learner.models.model,device_ids=[0, 1])

pretrained_model_name = '20181227135821_size512_B10_lr0.0001_se_resnext50_best' # 512
print(f'load pretrained model: {pretrained_model_name}')
learner.load(pretrained_model_name)
learner.set_data(md)


n_aug=16
if args.debug:
    print('run debug mode')
    n_aug=2
    
# small batchsize
#bs = 8
#md = get_data(sz,bs)
#learner.set_data(md)

preds,y = learner.TTA(n_aug=n_aug)
preds = np.stack(preds, axis=-1)
preds = utils_pytorch.sigmoid_np(preds)
#pred = preds.max(axis=-1)
pred = preds.mean(axis=-1)


th = utils_pytorch.fit_val(pred,y)
th[th<0.1] = 0.1
print('Thresholds: ',th)
f1_macro_score = f1_score(y, pred>th, average='macro')
print('F1 macro: ',f1_macro_score)
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th, average='micro'))

print('Fractions: ',(pred > th).mean(axis=0))
print('Fractions (true): ',(y > th).mean(axis=0))

# print('search thresholds by cross valdiation')
# th, score, cv = 0,0,10
# for i in range(cv):
#     xt,xv,yt,yv = train_test_split(pred,y,test_size=0.5,random_state=i)
#     th_i = utils_pytorch.fit_val(xt,yt)
#     th += th_i
#     score += f1_score(yv, xv>th_i, average='macro')
# th/=cv
# score/=cv
# print('Thresholds: ',th)
# print('F1 macro avr:',score)
# print('F1 macro: ',f1_score(y, pred>th, average='macro'))
# print('F1 micro: ',f1_score(y, pred>th, average='micro'))

# stats = np.array([[0.05908022, 0.04532852, 0.04065233, 0.05923426], [0.10371015, 0.07984633, 0.10664798, 0.09878183]])
# md = get_data(sz,bs, stats)
# learner.set_data(md)

preds_t,y_t = learner.TTA(n_aug=n_aug,is_test=True)
preds_t = np.stack(preds_t, axis=-1)
preds_t = utils_pytorch.sigmoid_np(preds_t)
print(preds_t.shape)
#pred_t = preds_t.max(axis=-1) #max works better for F1 macro score
pred_t = preds_t.mean(axis=-1) #max works better for F1 macro score
print(pred_t.shape)

    
# utils_pytorch.save_pred(learner, pred_t,th_t, f'protein_classification_val{f1_macro_score:.4f}.csv')
utils_pytorch.save_pred(learner, pred_t,th, f'protein_classification_val{f1_macro_score:.4f}.csv')

th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])
print('Fractions: ',(pred_t > th_t).mean(axis=0))
print('Thresholds: ',th_t)
f1_macro_score_manual = f1_score(y, pred>th_t, average='macro')
print('F1 macro: ',f1_macro_score_manual)
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th_t, average='micro'))
utils_pytorch.save_pred(learner, pred_t,th_t, f'protein_classification_manual_th_val{f1_macro_score_manual:.4f}.csv')

lb_prob = [
 0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
 0.075268817,0.043841336,0.075268817,0.010000000,0.010000000,
 0.010000000,0.043841336,0.043841336,0.014198783,0.043841336,
 0.010000000,0.028806584,0.014198783,0.028806584,0.059322034,
 0.010000000,0.126126126,0.028806584,0.075268817,0.010000000,
 0.222493880,0.028806584,0.010000000]
# I replaced 0 by 0.01 since there may be a rounding error leading to 0


th_t = utils_pytorch.fit_test(pred_t,lb_prob)
th_t[th_t<0.1] = 0.1
print('Thresholds: ',th_t)
print('Fractions: ',(pred_t > th_t).mean(axis=0))
print('Fractions (th = 0.5): ',(pred_t > 0.5).mean(axis=0))

print('Thresholds: ',th_t)
print('F1 macro: ',f1_score(y, pred>th_t, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th_t, average='micro'))

utils_pytorch.save_pred(learner, pred_t,th_t,'protein_classification_f.csv')

utils_pytorch.save_pred(learner, pred_t,th,'protein_classification_v.csv')
utils_pytorch.save_pred(learner, pred_t,0.5,'protein_classification_05.csv')

class_list = [8,9,10,15,20,24,27]
for i in class_list:
    th_t[i] = th[i]
utils_pytorch.save_pred(learner, pred_t,th_t,'protein_classification_c.csv')

labels = pd.read_csv(utils_pytorch.LABELS).set_index('Id')
label_count = np.zeros(len(utils_pytorch.name_label_dict))
for label in labels['Target']:
    l = [int(i) for i in label.split()]
    label_count += np.eye(len(utils_pytorch.name_label_dict))[l].sum(axis=0)
label_fraction = label_count.astype(np.float)/len(labels)
label_count, label_fraction

th_t = utils_pytorch.fit_test(pred_t,label_fraction)
th_t[th_t<0.05] = 0.05
print('Thresholds: ',th_t)
print('Fractions: ',(pred_t > th_t).mean(axis=0))
utils_pytorch.save_pred(learner, pred_t,th_t,'protein_classification_t.csv')

print('Thresholds: ',th_t)
print('F1 macro: ',f1_score(y, pred>th_t, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th_t, average='micro'))







