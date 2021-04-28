#!/usr/bin/env python
# coding: utf-8

# In[3]:


# change path
code_path = 'Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/'
label_save_path = 'Data_ham/HAM_LABELS.npy'

img_path1 = 'Data_ham/HAM10000_images_part_1/'
img_path2 = 'Data_ham/HAM10000_images_part_2/'
# gt_path = 'Data_ham/HAM_MASK/'
gt_path = 'Data_ham/HAM10000_segmentations_lesion_tschandl/'

el_path = 'Data_ham/HAM_ellipse/'
grab_path = 'Data_ham/HAM_grabcut/'

a1_path='checkpoints/ham_a1.pt'
a1_class_path='checkpoints/ham_a1class.pt'

num_class=45


# In[4]:


import torch, math, time
import numpy as np
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


# ## Dataloader

# In[5]:


im_dir = np.array(sorted(os.listdir(img_path1)+os.listdir(img_path2)))
gt_dir = np.array(sorted(os.listdir(gt_path)))
el_dir = np.array(sorted(os.listdir(el_path)))
grab_dir = np.array(sorted(os.listdir(grab_path)))

wrong=[1736, 3925, 5513]
im_dir=np.delete(im_dir,wrong)
gt_dir=np.delete(gt_dir,wrong)

labels=np.load(label_save_path)


# In[6]:


random.seed(1)
ind = np.arange(len(im_dir))
shuffle(ind)
tr = int(len(ind)*0.7)
val = tr+int(len(ind)*0.2)

im_train_dir = im_dir[ind[:tr]]
im_val_dir = im_dir[ind[tr:val]]
im_test_dir = im_dir[ind[val:]]


# In[7]:


random.seed(1)
ind = np.arange(len(im_dir))
shuffle(ind)
tr = int(len(ind)*0.7)
val = tr+int(len(ind)*0.2)
im_train_dir = im_dir[ind[:tr]]
gt_train_dir = gt_dir[ind[:tr]]
el_train_dir = el_dir[ind[:tr]]
grab_train_dir = grab_dir[ind[:tr]]
train_label=labels[ind[:tr]]

im_val_dir = im_dir[ind[tr:val]]
gt_val_dir = gt_dir[ind[tr:val]]
el_val_dir = el_dir[ind[tr:val]]
grab_val_dir = grab_dir[ind[tr:val]]
val_label=labels[ind[tr:val]]

im_test_dir = im_dir[ind[val:]]
gt_test_dir = gt_dir[ind[val:]]
el_test_dir = el_dir[ind[val:]]
grab_test_dir = grab_dir[ind[val:]]
test_label=labels[ind[val:]]


# In[8]:


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


# In[9]:


class Dataset(Dataset):
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            masks_path,
            label,
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = []
        for image_id in images_dir:
            if os.path.isfile(os.path.join(img_path1, image_id)):
                self.images_fps.append(os.path.join(img_path1, image_id))
            else:
                self.images_fps.append(os.path.join(img_path2, image_id))
        self.masks_fps = [os.path.join(masks_path, mask_id) for mask_id in masks_dir]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.label=label
    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)/255.0
        mask[mask>=0.5]=1
        mask[mask<0.5]=0
        mask = np.expand_dims(mask,2)
        sample = self.augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].reshape(384,480,1)
        sample = self.preprocessing(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
        return image, mask, torch.tensor(self.label[i])
        
    def __len__(self):
        return len(self.masks_fps)


# In[10]:


propro=get_preprocessing(get_preprocessing_fn('resnet101', pretrained='imagenet'))
train_dataset = Dataset(
    im_train_dir, 
    el_train_dir, #grab_train_dir, 
    el_path, #grab_path
    train_label,
    augmentation=get_training_augmentation(),
    preprocessing=propro
)
val_dataset = Dataset(
    im_val_dir, 
    el_val_dir, #grab_val_dir, 
    el_path, #grab_path,
    val_label,
    augmentation=get_training_augmentation(),
    preprocessing=propro
)
test_dataset = Dataset(
    im_test_dir, 
    gt_test_dir, 
    gt_path,
    test_label,
    augmentation=get_training_augmentation(),
    preprocessing=propro
)


# In[11]:


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



# ## A1+class Train

# In[ ]:


def train(model, opt, criterion_grab, criterion_class, device, checkpoint_save_path, max_patient=5):
    min_val_loss = float('inf')
    m=0
    for epoch in tqdm(range(50)):
        # train
        model.train()
        train_loss={'Total':[],'Mask':[],'Label':[]}
        for batch_idx, sam in enumerate(train_loader):
            # send to device
            data, label, grabcut = sam[0].to(device=device,dtype=torch.float), sam[2].to(device=device,dtype=torch.long), sam[1].to(device=device,dtype=torch.float)
            
            opt.zero_grad()
            
            out_grab, out_class = model(data)
        
            loss_grab = criterion_grab(out_grab, grabcut) * 50.0
            loss_class = criterion_class(out_class, label)
            loss = loss_grab+loss_class
            loss.backward()
            opt.step()
            
            train_loss['Total'].append(loss.item())
            train_loss['Mask'].append(loss_grab.item())
            train_loss['Label'].append(loss_class.item())
            
            if batch_idx % 500 == 0:
                total_loss = sum(train_loss['Total'])/len(train_loss['Total'])
                mask_loss = sum(train_loss['Mask'])/len(train_loss['Mask'])
                label_loss = sum(train_loss['Label'])/len(train_loss['Label'])
                print('Step %s avg train total loss = %.4f, avg train Mask loss = %.4f, avg train label loss = %.4f,'%(batch_idx, total_loss, mask_loss, label_loss))
                train_loss = {'Total':[],'Mask':[],'Label':[]}
                
        # valid
        valid_loss={'Total':[],'Mask':[],'Label':[]}
        model.eval()
        with torch.no_grad():
            for batch_idx, sam in enumerate(val_loader):
                data, label, grabcut = sam[0].to(device=device,dtype=torch.float), sam[2].to(device=device,dtype=torch.long), sam[1].to(device=device,dtype=torch.float)
                
                out_grab,out_class=model(data)
                
                loss_grab = criterion_grab(out_grab,grabcut) * 50.0
                loss_class = criterion_class(out_class, label)
                loss = loss_grab+loss_class
                
                valid_loss['Total'].append(loss.item())
                valid_loss['Mask'].append(loss_grab.item())
                valid_loss['Label'].append(loss_class.item())
                
            total_loss = sum(valid_loss['Total'])/len(valid_loss['Total'])
            mask_loss = sum(valid_loss['Mask'])/len(valid_loss['Mask'])
            label_loss = sum(valid_loss['Label'])/len(valid_loss['Label'])
            print('At Epoch %s avg validation total loss = %.4f, Mask loss = %.4f, label loss = %.4f,'%(epoch, total_loss, mask_loss, label_loss))
        
        if mask_loss < min_val_loss:
            min_val_loss = mask_loss
            torch.save({'model_dict': model.state_dict()},checkpoint_save_path)
            print('model saved')
            m=0
        else:
            m+=1
            if m >= max_patient:
                torch.save({'model_dict': model.state_dict()},checkpoint_save_path)
                return
                
    return


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr=0.001 

model = smp.DeepLabV3Plus(encoder_name='resnet101',aux_params={'classes':num_class})
model = model.to(device)

opt = Adam(model.parameters(),lr=lr)

bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

train(model,opt,bce_loss,ce_loss,device, a1_class_path, max_patient=5)
print('A1+class training completed.')





