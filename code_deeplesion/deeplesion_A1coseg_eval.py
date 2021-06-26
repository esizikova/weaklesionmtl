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
acoseg_path = 'checkpoints/checkpoint_coseg_'+str(num_class)+'.pt'

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
from util import metrics
#from livelossplot import PlotLosses

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

raw_images=np.array(raw_images)

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
from matplotlib import image
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


BATCH_SIZE = 8                                            
train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=1, drop_last=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=1,drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=1,drop_last=True)

new_coseg_model = model1().cuda()
checkpoint = torch.load(acoseg_path)
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
        data = sam['image'].float().cuda()
        target = sam['mask'].float().cuda()
        
        output1, output2 = new_coseg_model(data[0:4], data[4:8])
        output1 = sig(output1)
        output2 = sig(output2)
               
        loss = criterion(output1.squeeze(), target[0:4].squeeze(1)) + criterion(output2.squeeze(), target[4:8].squeeze(1))
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