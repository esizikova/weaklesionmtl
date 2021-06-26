#!/usr/bin/env python
# coding: utf-8

img_path='Data_deep/Images_png/'
train_idx_path='Data_deep/train.npy'
val_idx_path='Data_deep/val.npy'
test_idx_path='Data_deep/test.npy'
train_grab_path='Data_deep/train_grabcut.npy'
val_grab_path='Data_deep/val_grabcut.npy'
test_grab_path='Data_deep/test_grabcut.npy'
el_path='Data_deep/mask/'
recist_path='Data_deep/ell.npy'
code_path='Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/'
num_class=190
a1_path = 'checkpoints/checkpoint.pt'
a1_class_path = 'checkpoints/checkpoint_'+str(num_class)+'.pt'

import torch, math, time
import numpy as np
from sklearn.cluster import KMeans
import cv2
import PIL
from matplotlib import pyplot as plt
from matplotlib import image
import skimage, skimage.transform
import sys
sys.path.append(code_path)
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
import albumentations as albu
from tqdm.auto import tqdm
import torch.nn as nn
import torchvision.models.resnet as resnet_util
from torch.optim import Adam
import os
from copy import deepcopy
import random
from random import shuffle
from util import metrics
from model import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


# train_val_test index
train_idx = np.load(train_idx_path)
val_idx = np.load(val_idx_path)
test_idx = np.load(test_idx_path)

delete_idx = [24897,3136,5780,9239,14485,29866,14474,6768]
train_idx = np.setdiff1d(train_idx,delete_idx)
test_idx = np.setdiff1d(test_idx,delete_idx)
val_idx = np.setdiff1d(val_idx,delete_idx)


# preprocessed image
raw_images = []
for i in range(1,5):
    data_path = img_path+'images_total_'+str(i)+'.mat'
    data = sio.loadmat(data_path)
    NIMAGES = data['image'].shape[0] 
    raw_images+=[data['image'][index] for index in range(NIMAGES)]

raw_images = np.array(raw_images)
train_raw = raw_images[train_idx]
val_raw = raw_images[val_idx]
test_raw = raw_images[test_idx]
train_raw = [patch.astype(float) / 255 for patch in train_raw]
test_raw = [patch.astype(float) / 255 for patch in test_raw]
val_raw = [patch.astype(float) / 255 for patch in val_raw]

# grabcut mask
train_grab = np.load(train_grab_path)
val_grab = np.load(val_grab_path)
test_grab = np.load(test_grab_path)

# ellipse mask
mask = [image.imread(el_path+str(i)+'.jpg')[:,:,0] for i in range(len(raw_images))]
mask = np.stack(mask)

m = mask/255.0
mas = np.ma.masked_where((m>0.9),m)
mas = mas.filled(fill_value=1)
mas = np.ma.masked_where((mas!=1),mas)
mas = mas.filled(fill_value=0)
mas = np.uint8(mas)

train_ell=np.array(mas[train_idx])
val_ell=np.array(mas[val_idx])
test_ell=np.array(mas[test_idx])

# Class Label
recist_ = np.load(recist_path)
km=KMeans(n_clusters=num_class, random_state=0).fit(recist_)
labels = km.labels_

train_labels = labels[train_idx]
val_labels = labels[val_idx]
test_labels = labels[test_idx]

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class WSOLDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, label, grabcut, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grabcut = grabcut
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = augmentation(image=self.data[idx],mask=self.grabcut[idx])
        sample1 = preprocessing(image=sample['image'],mask=sample['mask'].reshape(384,480,1))
        out = {'image':sample1['image'],'mask':sample1['mask'],'label':torch.tensor(self.label[idx])}
        return out

preprocess_input = get_preprocessing_fn('resnet101', pretrained='imagenet')
augmentation = get_validation_augmentation()
preprocessing = get_preprocessing(preprocess_input)

train_dataset = WSOLDataset(data=train_raw,label=train_labels,grabcut=train_ell)
val_dataset = WSOLDataset(data=val_raw,label=val_labels,grabcut=val_ell)
test_dataset = WSOLDataset(data=test_raw,label=test_labels,grabcut=test_grab)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=10, shuffle=True,
                                             num_workers=1)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=10, shuffle=True,
                                             num_workers=1)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=10, shuffle=False,
                                             num_workers=1)

# ## A1 Train
def train(model, opt, criterion, device, checkpoint, max_patient=5):
    min_val_loss = float('inf')
    m=0
    for epoch in tqdm(range(50)):
        # train
        model.train()
        train_loss=[]
        for batch_idx, sam in enumerate(train_loader):
            # send to device
            data, target = sam['image'].to(device=device,dtype=torch.float), sam['mask'].to(device=device,dtype=torch.float)
            
            opt.zero_grad()
            
            out=model(data)
        
            loss = criterion(out, target) * 100.0
            loss.backward()
            opt.step()
            
            train_loss.append(loss.item())
            
            if batch_idx % 500 == 0:
                avg_loss = sum(train_loss)/len(train_loss)
                print('Step {} avg train loss = {:.{prec}f}'.format(batch_idx, avg_loss, prec=4))
                train_loss = []
                
        # valid
        valid_loss=[]
        model.eval()
        with torch.no_grad():
            for batch_idx, sam in enumerate(val_loader):
                data, target = sam['image'].to(device=device,dtype=torch.float), sam['mask'].to(device=device,dtype=torch.float)
                
                out=model(data)
                
                loss = criterion(out,target) * 100.0
                valid_loss.append(loss.item())
            avg_val_loss = sum(valid_loss) / len(valid_loss)
            print('Validation loss after {} epoch = {:.{prec}f}'.format(epoch, avg_val_loss, prec=4))
        
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            torch.save({'model_dict': model.state_dict()},checkpoint)
            print('model saved')
            m=0
        else:
            m+=1
            if m >= max_patient:
                return
                
    return 




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr=0.001 

model = smp.DeepLabV3Plus(encoder_name='resnet101')
model = model.to(device)

opt = Adam(model.parameters(), lr=lr)
bce_loss = nn.BCEWithLogitsLoss()

train(model,opt,bce_loss,device,a1_path)
