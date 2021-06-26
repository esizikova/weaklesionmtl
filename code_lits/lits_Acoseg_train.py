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


# ### Train
def get_dice_iou(pred, mas, iou, dice):
    # iou, dice = [], []
    for i in range(len(pred)):
        pre = pred[i,132:252,180:300].numpy()
        gt = mas[i,132:252,180:300].numpy()
        inter = np.logical_and(pre==1,gt==1)
        union  = np.logical_or(pre==1,gt==1)
        iou.append(np.sum(inter)/np.sum(union))
        dice.append(np.sum(pre[gt==1])*2.0 / (np.sum(pre) + np.sum(gt)))

def train(epoch):
    for batch_idx, sam in enumerate(train_loader):
        data = sam[0].float().cuda()
        target = sam[1].float().cuda()

        data_upsampled = data
        target_upsampled = target

        # go pair by pair
        sub_batch_id = 0
        for ind in [i*COSEG_BATCH_SIZE for i in range(int(data_upsampled.shape[0]/COSEG_BATCH_SIZE))]:
            b1 = (ind,ind+COSEG_BATCH_SIZE)
            if ind+COSEG_BATCH_SIZE >= data_upsampled.shape[0]:
                b2 = (0,COSEG_BATCH_SIZE)
            else:
                b2 = (ind+COSEG_BATCH_SIZE,ind+COSEG_BATCH_SIZE*2)

            output1, output2 = new_coseg_model(data_upsampled[b1[0]:b1[1]], data_upsampled[b2[0]:b2[1]])

            loss1 = criterion(sig(output1.squeeze()), target_upsampled[b1[0]:b1[1]].squeeze(1))
            loss2 = criterion(sig(output2.squeeze()), target_upsampled[b2[0]:b2[1]].squeeze(1))

            loss = loss1 + loss2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(new_coseg_model.parameters(), 0.05)
            optimizer.step()

            out1 = sig(output1.squeeze()).cpu()
            out2 = sig(output2.squeeze()).cpu()

            pred1 = torch.empty(out1.shape)
            pred2 = torch.empty(out2.shape)
            pred1[out1>=0.5]=1
            pred1[out1<0.5]=0
            pred2[out2>=0.5]=1
            pred2[out2<0.5]=0

            iou, dice = [], []
            get_dice_iou(pred1, target_upsampled[b1[0]:b1[1]].squeeze(1).long().cpu(), iou, dice)
            get_dice_iou(pred2, target_upsampled[b2[0]:b2[1]].squeeze(1).long().cpu(), iou, dice)

            avg_iou=np.mean(iou)
            avg_dice=np.mean(dice)

            BATCH_ID = batch_idx * N_SUB_BATCHES + sub_batch_id
            print("Epoch %s: Batch %i/%i Loss %.2f iou %.2f dice %.2f" % (epoch, BATCH_ID,N_BATCHES, loss.item(), avg_iou, avg_dice))
            
def valid(epoch):
    val_loss=[]
    dice_list = []
    with torch.no_grad():
        for batch_idx, sam in enumerate(val_loader):
            data = sam[0].float().cuda()
            target = sam[1].float().cuda()

            data_upsampled = data
            target_upsampled = target

            # go pair by pair
            sub_batch_id = 0
            for ind in [i*COSEG_BATCH_SIZE for i in range(int(data_upsampled.shape[0]/COSEG_BATCH_SIZE))]:
                b1 = (ind,ind+COSEG_BATCH_SIZE)
                if ind+COSEG_BATCH_SIZE >= data_upsampled.shape[0]:
                    b2 = (0,COSEG_BATCH_SIZE)
                else:
                    b2 = (ind+COSEG_BATCH_SIZE,ind+COSEG_BATCH_SIZE*2)

                output1, output2 = new_coseg_model(data_upsampled[b1[0]:b1[1]], data_upsampled[b2[0]:b2[1]])

                loss1 = criterion(sig(output1.squeeze()), target_upsampled[b1[0]:b1[1]].squeeze(1))
                loss2 = criterion(sig(output2.squeeze()), target_upsampled[b2[0]:b2[1]].squeeze(1))

                loss = loss1 + loss2
                val_loss.append(loss.item())
                
                out1 = sig(output1.squeeze()).cpu()
                out2 = sig(output2.squeeze()).cpu()

                pred1 = torch.empty(out1.shape)
                pred2 = torch.empty(out2.shape)
                pred1[out1>=0.5]=1
                pred1[out1<0.5]=0
                pred2[out2>=0.5]=1
                pred2[out2<0.5]=0

                iou, dice = [], []
                get_dice_iou(pred1, target_upsampled[b1[0]:b1[1]].squeeze(1).long().cpu(), iou, dice)
                get_dice_iou(pred2, target_upsampled[b2[0]:b2[1]].squeeze(1).long().cpu(), iou, dice)
                dice_list.append(np.mean(dice))

    avg_loss=np.mean(val_loss)
    avg_dice=np.mean(dice_list)
    print('Val loss for epoch %s is'%(epoch),round(100*avg_loss,4))
    print('Val dice for epoch %s is'%(epoch),round(100*avg_dice,4))
    return avg_dice


model_save_path = 'checkpoint/LITS_Acoseg.pt'
criterion = nn.BCELoss()
criterion.cuda()

new_coseg_model = model1().cuda()
    
sig = nn.Sigmoid()
optimizer = Adam(new_coseg_model.parameters(), lr=1e-5)

COSEG_BATCH_SIZE = 4
N_LARGE_BATCHES = len(train_loader)
BATCH_SIZE = 8
N_SUB_BATCHES = BATCH_SIZE // COSEG_BATCH_SIZE

N_BATCHES = N_SUB_BATCHES * N_LARGE_BATCHES
  
max_val_dice=-1
max_patient=4
m=0
for epoch in range(50):
    print('Epoch ', epoch)
    new_coseg_model.train()
    train(epoch)
    new_coseg_model.eval()
    new_val_dice = valid(epoch)
    if new_val_dice > max_val_dice:
        max_val_dice = new_val_dice
        torch.save({'model_dict': new_coseg_model.state_dict()},model_save_path)
        print('model saved')
        m=0
    else:
        m+=1
        if m >= max_patient:
            break
print('Training completed')

