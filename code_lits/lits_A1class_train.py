#!/usr/bin/env python
# coding: utf-8

import torch, math, time
import numpy as np
import cv2
import PIL
import numpy as np
from matplotlib import image
import skimage, skimage.transform
import time
import sys
sys.path.append('segmentation_models.pytorch/')
#sys.path.append('Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/')
import scipy.io as sio
from scipy.io import loadmat
from numpy import asarray
import torch.nn.functional as F
# CUDA flag. Speed-up due to CUDA is mostly noticable for large batches.
cuda = True
from PIL import Image,ImageDraw
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from skimage import io, transform
from model import *
from torchvision.utils import save_image
import segmentation_models_pytorch as smp
import albumentations as albu
from tqdm.auto import tqdm
import torch.nn as nn
import torchvision.models.resnet as resnet_util
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.optim import Adam
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR
import random
from random import shuffle
from segmentation_models_pytorch.base import ClassificationHead
import medpy.metric as mm

def get_training_augmentation():
    train_transform = [albu.Resize(120,120),albu.PadIfNeeded(384, 480)]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

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
            labels,
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = [os.path.join(im_path, image_id) for image_id in images_dir]
        self.masks_fps = [os.path.join(masks_path, mask_id) for mask_id in masks_dir]
        self.labels = labels
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
        return image, mask,self.labels[i]
        
    def __len__(self):
        return len(self.masks_fps)


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

def train(epoch, model,flag = 1):
    model.train()
    train_loss={'Total':[],'Mask':[],'Label':[]}
    for batch_idx, sam in enumerate(train_loader):
        # send to device
        data,target,label = sam[0].to(device=device,dtype=torch.float), sam[1].to(device=device,dtype=torch.float), sam[2].to(device=device,dtype=torch.long)
        adam_optimizer.zero_grad()
        pred,classified = model(data)
        grab_val = bce_loss(pred, target) 
        if flag==1:
            class_val = ce_loss(classified, label)
            loss_val = weight*grab_val+class_val
            loss_val.backward()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\t MaskLoss: {:.6f}\tLabelLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                     loss_val.item(),100*grab_val.item(),class_val.item()),flush=True)
            train_loss['Total'].append(loss_val.item())
            train_loss['Mask'].append(100*grab_val.item())
            train_loss['Label'].append(class_val.item())
        else:
            grab_val.backward()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                     100*grab_val.item()),flush=True)
            train_loss['Mask'].append(100*grab_val.item())
            
        adam_optimizer.step()
    print("Average Train Loss:","Total",round(np.mean(train_loss['Total'][-100:]),4),
            "Mask",round(np.mean(train_loss['Mask'][-100:]),4),
            "Label",round(np.mean(train_loss['Label'][-100:]),4),flush=True)
    


def validate(model,model_name,max_val_dice,flag,m):
    valid_loss={'Total':[],'Mask':[],'Label':[]}
    model.eval()
    grabcut = []
    pred_list = []
    dice = []
    with torch.no_grad():
        for batch_idx, sam in enumerate(val_loader):
            data,target,label = sam[0].to(device=device,dtype=torch.float), sam[1].to(device=device,dtype=torch.float), sam[2].to(device=device,dtype=torch.long)
            pred,classified = model(data)
            grabcut.append(target.cpu().detach().numpy())
            pred_list.append(pred.cpu().detach().numpy())
            grab_val = bce_loss(pred, target) 
            if flag == 1:
                class_val = ce_loss(classified, label)
                loss = weight*grab_val+class_val
                valid_loss['Total'].append(loss.item())
                valid_loss['Mask'].append(100*grab_val.item())
                valid_loss['Label'].append(class_val.item())
            else:
                valid_loss['Mask'].append(grab_val.item())
                
        gt_map = np.concatenate(grabcut, axis=0)
        pred_map = np.concatenate(pred_list, axis=0)
        pred_map_sig = nn.Sigmoid()(torch.tensor(pred_map))
        threshold = 0.5
        predict = deepcopy(pred_map_sig)
        predict[pred_map_sig>=threshold]=1
        predict[pred_map_sig<threshold]=0
        cut_pred = []
        cut_gt = []
        for j in range(predict.shape[0]):
            cut_pred.append(np.uint8(predict[j,0,132:252,180:300]))
            cut_gt.append(np.uint8(gt_map[j,0,132:252,180:300]))
        for i in range(len(cut_pred)):
            mas_ = cut_gt[i]
            pred_ = cut_pred[i]
            dice.append(np.sum(pred_[mas_==1])*2.0 / (np.sum(pred_) + np.sum(mas_)))
    
    dice_mean = np.mean(dice)
    print("Average Validation Loss:",
              "Total", round(np.mean(valid_loss['Total']),4),
              "Mask", round(np.mean(valid_loss['Mask']),4),
              "Label", round(np.mean(valid_loss['Label']),4),"Dice",round(dice_mean,4),flush=True) 
    
    if dice_mean > max_val_dice:
        max_val_dice = dice_mean
        torch.save({'model_dict': model.state_dict()},model_name)
        print('model saved',flush=True)
        m=0
    else:
        m+=1
    return max_val_dice,m

weight = 50
im_path = 'data/newimg_patch/'
gt_path = 'data/newgt_patch/'
el_path = 'data/newellipse/'
gr_path = 'data/newgrab/'
im_dir = np.array(sorted(os.listdir(im_path)))
gt_dir = np.array(sorted(os.listdir(gt_path)))
el_dir = np.array(sorted(os.listdir(el_path)))
gr_dir = np.array(sorted(os.listdir(gr_path)))
#im_dir = np.array([item for item in im_dir if item[-3:]!='npy'])
#gt_dir = np.array([item for item in gt_dir if item[-3:]!='npy'])

#tr,val,te = list(np.arange(int(len(im_dir)*0.7))), list(np.arange(int(len(im_dir)*0.7), int(len(im_dir)*0.85))), list(np.arange(int(len(im_dir)*0.85),len(im_dir)))
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

print("train, val, test data size",len(im_train_dir),len(im_val_dir),len(im_test_dir))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ellipses = np.load('data/ellipses_total.npy')
clusters = [40]
for i in range(len(clusters)):
    model_name = 'checkpoint/LiTS_A1class_'+str(clusters[i])+'.pt'
    kmeans = KMeans(n_clusters=clusters[i], random_state=0).fit(ellipses)
    labels = kmeans.labels_
    labels = torch.tensor(labels)
    propro=get_preprocessing(get_preprocessing_fn('resnet101', pretrained='imagenet'))
    train_dataset = Dataset(
        im_train_dir, 
        el_train_dir, 
        el_path,
        labels[tr],
        augmentation=get_training_augmentation(),
        preprocessing=propro
    )
    val_dataset = Dataset(
        im_val_dir, 
        el_val_dir, 
        el_path,
        labels[val],
        augmentation=get_training_augmentation(),
        preprocessing=propro
    )
    test_dataset = Dataset(
        im_test_dir, 
        gt_test_dir, 
        gt_path,
        labels[te],
        augmentation=get_training_augmentation(),
        preprocessing=propro
    )
    bs = 10
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=bs, shuffle=True,
                                                 num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=bs, shuffle=True,
                                                 num_workers=1)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=bs, shuffle=False,
                                                 num_workers=1)
    
    model = smp.DeepLabV3Plus(encoder_name='resnet101').to(device)
    model.classification_head = ClassificationHead(in_channels=model.encoder.out_channels[-1], classes=clusters[i])
    model = model.to(device)

    # Fine tune from the scratch 
    lr = 0.001
    max_val_dice = -1
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    adam_optimizer = Adam(model.parameters(),lr=lr)#,weight_decay=1e-6)
    m=0
    s = time.time()
    for epoch in range(50):
        train(epoch, model,1)
        max_val_dice,m = validate(model,model_name,max_val_dice,1,m)
        if m>=5:
            break
    e = (time.time()-s)/60
    print("Time taken:",e,flush=True)
    print('Training completed.')

