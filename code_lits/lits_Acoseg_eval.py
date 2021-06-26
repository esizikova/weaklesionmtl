#!/usr/bin/env python
# coding: utf-8

# set path
im_path = 'data/newimg_patch/'
gt_path = 'data/newgt_patch/'
el_path = 'data/newellipse/'
gr_path = 'data/newgrab/'
import torch, math, time
import numpy as np
import cv2
import PIL
from matplotlib import pyplot as plt
from matplotlib import image
import skimage, skimage.transform
import sys
from torchvision import models
import scipy.io as sio
from scipy.io import loadmat
import torch.nn.functional as F
# CUDA flag. Speed-up due to CUDA is mostly noticable for large batches.
cuda = True
from PIL import Image,ImageDraw
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Scale, ToPILImage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from skimage import io, transform
from model import *
import segmentation_models_pytorch as smp
import albumentations as albu
from tqdm.auto import tqdm
import torch.nn as nn
import torchvision.models.resnet as resnet_util
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.optim import Adam
import os
from copy import deepcopy
import random
from random import shuffle
import medpy.metric as mm
# Evaluation Metrics
from skimage import feature

def get_center(mask):
    cand=np.where(mask==1)
    if len(cand[1])==0:
        return [0,0]
    x_min=min(cand[1])
    x_max=max(cand[1])
    y_min=min(cand[0])
    y_max=max(cand[0])
    center = [(x_min+x_max)/2, (y_min+y_max)/2]
    return center

def getcircum(mask):
    edg = feature.canny(mask,low_threshold=1,high_threshold=1)
    cand = np.where(edg==True)
    return len(cand[0])

def metrics(mas,pred):
    cir, dis = [], []
    iou, dice = [], []
    avd, vs = [], []
    for i in range(len(pred)):
        mas_ = mas[i]
        pred_ = pred[i]

        result = np.atleast_1d(pred_.astype(np.bool))
        reference = np.atleast_1d(mas_.astype(np.bool))
        tp = np.count_nonzero(result & reference)
        fn = np.count_nonzero(~result & reference)
        fp = np.count_nonzero(result & ~reference)
        vs.append(1 - abs(fn-fp)/(2*tp+fp+fn))

        true = getcircum(mas_)
        pre = getcircum(pred_)
        cir.append(abs(true-pre))

        inter = np.logical_and(pred_==1,mas_==1)
        union  = np.logical_or(pred_==1,mas_==1)
        iou.append(np.sum(inter)/np.sum(union))
        dice.append(np.sum(pred_[mas_==1])*2.0 / (np.sum(pred_) + np.sum(mas_)))
        if (pred_==0).all():
            avd.append(float('-inf'))
            dis.append(float('-inf'))
        else:
            avd.append(mm.binary.hd(mas_,pred_))
            true = np.array(get_center(mas_))
            pre = np.array(get_center(pred_))
            dis.append(np.linalg.norm(true-pre))

    dm=max(dis)
    am=max(avd)
    dis = [dm if item == float('-inf') else item for item in dis]
    avd = [am if item == float('-inf') else item for item in avd]
    print("Best test IOU is", np.mean(iou))
    print("Best test DICE is", np.mean(dice))
    print('Test center error is', np.mean(dis))
    print('Test circumstance error is', np.mean(cir)/2)
    print('Test AVD is ', np.mean(avd))
    print('Test VS is ', np.mean(vs))



# ### Dataloader

im_dir = np.array(sorted(os.listdir(im_path)))
gt_dir = np.array(sorted(os.listdir(gt_path)))
el_dir = np.array(sorted(os.listdir(el_path)))
gr_dir = np.array(sorted(os.listdir(gr_path)))

#im_dir = np.array([item for item in im_dir if item[-3:]!='npy'])
#gt_dir = np.array([item for item in gt_dir if item[-3:]!='npy'])

#tr,val,te = list(np.arange(int(len(im_dir)/3))), list(np.arange(int(len(im_dir)/3), int(len(im_dir)*2/3))), list(np.arange(int(len(im_dir)*2/3),len(im_dir)))
tr,val,te = np.load('tr_ind.npy'),np.load('val_ind.npy'),np.load('test_ind.npy')
im_train_dir = im_dir[tr]
gt_train_dir = gt_dir[tr]
el_train_dir = el_dir[tr]
gr_train_dir = gr_dir[tr]
im_val_dir = im_dir[val]
gt_val_dir = gt_dir[val]
el_val_dir = el_dir[val]
gr_val_dir = gr_dir[val]
im_test_dir = im_dir[te]
gt_test_dir = gt_dir[te]
el_test_dir = el_dir[te]
gr_test_dir = gr_dir[te]



def get_training_augmentation():
    train_transform = [albu.Resize(120,120),albu.PadIfNeeded(384, 480)]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class Dataset(Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            masks_path,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_fps = [os.path.join(im_path, image_id) for image_id in images_dir]
        self.masks_fps = [os.path.join(masks_path, mask_id) for mask_id in masks_dir]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)/255.0
        mask = np.expand_dims(mask,2)
        sample = self.augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].reshape(384,480,1)
        sample = self.preprocessing(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
        return image, mask
        #return image, mask

    def __len__(self):
        return len(self.masks_fps)



propro=get_preprocessing(get_preprocessing_fn('resnet101', pretrained='imagenet'))

train_dataset = Dataset(
    im_train_dir,
    el_train_dir,  
    el_path,
    augmentation=get_training_augmentation(),
    preprocessing=propro
)
val_dataset = Dataset(
    im_val_dir,
    el_val_dir,  
    el_path,
    augmentation=get_training_augmentation(),
    preprocessing=propro
)

test_dataset = Dataset(
    im_test_dir,
    gt_test_dir,
    gt_path,
    augmentation=get_training_augmentation(),
    preprocessing=propro
)


bs = 8
train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=bs, shuffle=True,
                                             num_workers=1, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=bs, shuffle=True,
                                             num_workers=1, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=bs, shuffle=False,
                                             num_workers=1, drop_last=True)


model_save_path = 'checkpoint/LITS_Acoseg.pt'

COSEG_BATCH_SIZE = 4
N_LARGE_BATCHES = len(train_loader)
BATCH_SIZE = 8
N_SUB_BATCHES = BATCH_SIZE // COSEG_BATCH_SIZE

N_BATCHES = N_SUB_BATCHES * N_LARGE_BATCHES
  

# ### Evaluate
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_coseg_model = model1().to(device)
checkpoint = torch.load(model_save_path)
new_coseg_model.load_state_dict(checkpoint['model_dict'])
print('loaded pre-trained model')
sig = nn.Sigmoid()
criterion = nn.BCELoss()
test_loss=[]
mas = torch.tensor([])
res = torch.tensor([])

new_coseg_model.eval()
with torch.no_grad():
    for batch_idx, sam in enumerate(test_loader):
        #print(batch_idx)
        data, target = sam[0].to(device=device,dtype=torch.float), sam[1].to(device=device,dtype=torch.float)
        output1, output2 = new_coseg_model(data[0:COSEG_BATCH_SIZE], data[COSEG_BATCH_SIZE:BATCH_SIZE])
        output1 = sig(output1)
        output2 = sig(output2)
        loss = criterion(output1.squeeze(), target[0:COSEG_BATCH_SIZE].squeeze(1)) + criterion(output2.squeeze(), target[COSEG_BATCH_SIZE:BATCH_SIZE].squeeze(1))
        test_loss.append(loss.item())

        mas = torch.cat((mas,target.cpu()),dim=0)
        res = torch.cat((res,output1.cpu()),dim=0)
        res = torch.cat((res,output2.cpu()),dim=0)
    avg_test_loss = np.mean(test_loss)
    print('Test loss = {:.{prec}f}'.format(avg_test_loss, prec=4))

mas = np.array(mas)
res = np.array(res)
pred = deepcopy(res)
pred[res>=0.5]=1 #check threshold
pred[res<0.5]=0
cut_pred = []
cut_mas = []
for i in range(pred.shape[0]):
    cut_pred.append(np.uint8(pred[i,0,132:252,180:300]))
    cut_mas.append(np.uint8(mas[i,0,132:252,180:300]))
metrics(cut_mas,cut_pred)

