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
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
#from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from tensorboard_cb import TensorboardLogger
import torch
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
import utils_pytorch

parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument('--model', '-m', type=str, default='resnet34',
                    help='cnn model')
parser.add_argument('--weight', '-w', type=str, default=None,
                    help='pretrained model weight')
parser.add_argument('--batch', '-B', type=int, default=64,
                    help='batch size')
parser.add_argument('--size', '-s', type=int, default=256,
                    help='image size')
parser.add_argument('--lr', '-l', type=float, default=0.01,
                    help='learning rate')
args = parser.parse_args()


nw = 20   #number of workers for data loader
if args.model == 'resnet34':
    arch = resnet34 #specify target architecture
elif args.model == 'resnet50':
    arch = resnet50
else:
    raise ValueError('unknow archtecure')

train_names = list({f[:36] for f in os.listdir(utils_pytorch.TRAIN)})
test_names = list({f[:36] for f in os.listdir(utils_pytorch.TEST)})
# tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)

data_info = pd.read_csv(utils_pytorch.LABELS)
tr_n, val_n = train_test_split(data_info, test_size = 0.1, 
                 stratify = data_info['Target'].map(lambda x: x[:3] if '27' not in x else '0'), random_state=42)
tr_n = tr_n['Id'].tolist()
val_n = val_n['Id'].tolist()

def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(45, tfm_y=TfmType.NO),
                RandomFlip(),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    #mean and std in of each channel in the train set
    #stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
    #stats = A([0.0868 , 0.05959, 0.06522, 0.08891], [0.13044, 0.09792, 0.14862, 0.13281])
    #stats = A([0.0804419, 0.05262986, 0.05474701, 0.08270896], [0.13000701, 0.08796628, 0.1386317, 0.12718021]) # calulate myself
    stats = A([0.06734, 0.05087, 0.03266, 0.09257],[0.11997, 0.10335, 0.10124, 0.1574 ]) # include external data, caluculated by moriyama
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(utils_pytorch.pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],utils_pytorch.TRAIN), 
                (val_n,utils_pytorch.TRAIN), tfms, test=(test_names,utils_pytorch.TEST))
    md = ImageData(utils_pytorch.PATH, ds, bs, num_workers=nw, classes=None)
    return md

sz = args.size #image size
bs = args.batch  #batch size

print(f'image size: {sz}, batch size: {bs}, learning rate: {args.lr}')
dir_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S') + f'_size{sz}_B{bs}_lr{args.lr}_{args.model}'
if args.debug:
    dir_name = 'debug-' + dir_name
print(dir_name)
dir_path = os.path.join('test', dir_name)
best_model_path = dir_name + '_best_resnet'


md = get_data(sz,bs)
learner = utils_pytorch.ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
#pretrained_model_name = '20181129055744_size256best_resnet' # 256
#pretrained_model_name = '20181129122117best_resnet' # 299
if args.weight is not None:
    print('do transfer learning')
    print(f'pretrained model: {args.weight}')
    learner.load(args.weight)
    learner.set_data(md)

learner.opt_fn = optim.Adam
learner.clip = 1.0 #gradient clipping
learner.crit = utils_pytorch.FocalLoss()
learner.metrics = [utils_pytorch.acc, utils_pytorch.f1_torch]
tb_logger = TensorboardLogger(learner.model, md, dir_path, metrics_names=['acc', 'f1'])

lr = args.lr
learner.fit(lr,1, best_save_name=best_model_path, callbacks=[tb_logger])

if args.debug is False:
    learner.unfreeze()
    lrs=np.array([lr/10,lr/3,lr])
    learner.fit(lrs/4,4,cycle_len=2,use_clr=(10,20), best_save_name=best_model_path, callbacks=[tb_logger])

    learner.fit(lrs/4,2,cycle_len=4,use_clr=(10,20), best_save_name=best_model_path, callbacks=[tb_logger])

    learner.fit(lrs/16,1,cycle_len=8,use_clr=(5,20), best_save_name=best_model_path, callbacks=[tb_logger])

    learner.save('ResNet34_256_1')

learner.load(best_model_path)

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

if args.debug:
    n_aug=2
else:
    n_aug=16
    
# small batchsize
bs = 8
md = get_data(sz,bs)
learner.set_data(md)
preds,y = learner.TTA(n_aug=n_aug)
preds = np.stack(preds, axis=-1)
preds = sigmoid_np(preds)
pred = preds.max(axis=-1)

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y):
    params = 0.5*np.ones(len(utils_pytorch.name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

th = fit_val(pred,y)
th[th<0.1] = 0.1
print('Thresholds: ',th)
f1_macro_score = f1_score(y, pred>th, average='macro')
print('F1 macro: ',f1_macro_score)
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th, average='micro'))

print('Fractions: ',(pred > th).mean(axis=0))
print('Fractions (true): ',(y > th).mean(axis=0))


preds_t,y_t = learner.TTA(n_aug=n_aug,is_test=True)
preds_t = np.stack(preds_t, axis=-1)
preds_t = sigmoid_np(preds_t)
pred_t = preds_t.max(axis=-1) #max works better for F1 macro score

def save_pred(pred, th=0.5, fname='protein_classification.csv', use_leak=True):
    if use_leak:
        print('use leak')
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
        
    sample_df = pd.read_csv(utils_pytorch.SAMPLE)
    sample_list = list(sample_df.Id)
    leak_df = pd.read_csv('./data/test_matches.csv')
    pred_dic = {}
    for key, value in zip(learner.data.test_ds.fnames,pred_list):
        pred_dic[key] = value
        check_leak_df = leak_df.query('Test.str.contains(@key)' ,engine='python')
        if use_leak and len(check_leak_df) > 0:
            #print(f'found leak data ! key:{key}, target:{check_leak_df.iloc[0,5]}')
            pred_dic[key] = check_leak_df.iloc[0,5]
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(os.path.join('logs', dir_path, fname), header=True, index=False)
    
th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])
print('Fractions: ',(pred_t > th_t).mean(axis=0))
save_pred(pred_t,th_t, f'protein_classification_val{f1_macro_score:.4f}.csv')

print('Thresholds: ',th_t)
print('F1 macro: ',f1_score(y, pred>th_t, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th_t, average='micro'))

lb_prob = [
 0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
 0.075268817,0.043841336,0.075268817,0.010000000,0.010000000,
 0.010000000,0.043841336,0.043841336,0.014198783,0.043841336,
 0.010000000,0.028806584,0.014198783,0.028806584,0.059322034,
 0.010000000,0.126126126,0.028806584,0.075268817,0.010000000,
 0.222493880,0.028806584,0.010000000]
# I replaced 0 by 0.01 since there may be a rounding error leading to 0

def Count_soft(preds,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    return preds.mean(axis=0)

def fit_test(x,y):
    params = 0.5*np.ones(len(utils_pytorch.name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((Count_soft(x,p) - y,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

th_t = fit_test(pred_t,lb_prob)
th_t[th_t<0.1] = 0.1
print('Thresholds: ',th_t)
print('Fractions: ',(pred_t > th_t).mean(axis=0))
print('Fractions (th = 0.5): ',(pred_t > 0.5).mean(axis=0))

print('Thresholds: ',th_t)
print('F1 macro: ',f1_score(y, pred>th_t, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th_t, average='micro'))

save_pred(pred_t,th_t,'protein_classification_f.csv')

save_pred(pred_t,th,'protein_classification_v.csv')
save_pred(pred_t,0.5,'protein_classification_05.csv')

class_list = [8,9,10,15,20,24,27]
for i in class_list:
    th_t[i] = th[i]
save_pred(pred_t,th_t,'protein_classification_c.csv')

labels = pd.read_csv(utils_pytorch.LABELS).set_index('Id')
label_count = np.zeros(len(utils_pytorch.name_label_dict))
for label in labels['Target']:
    l = [int(i) for i in label.split()]
    label_count += np.eye(len(utils_pytorch.name_label_dict))[l].sum(axis=0)
label_fraction = label_count.astype(np.float)/len(labels)
label_count, label_fraction

th_t = fit_test(pred_t,label_fraction)
th_t[th_t<0.05] = 0.05
print('Thresholds: ',th_t)
print('Fractions: ',(pred_t > th_t).mean(axis=0))
save_pred(pred_t,th_t,'protein_classification_t.csv')

print('Thresholds: ',th_t)
print('F1 macro: ',f1_score(y, pred>th_t, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th_t, average='micro'))
