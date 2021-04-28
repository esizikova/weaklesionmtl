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



# In[12]:


def test(model,loader):
    test_loss=[]
    #imgs = torch.tensor([])
    mask,pred = [],[]
    with torch.no_grad():
        for batch_idx, sam in enumerate(loader):
            data, target = sam[0].to(device=device,dtype=torch.float), sam[1].to(device=device,dtype=torch.float)
            out = model(data)
            loss = bce_loss(out,target) * 100.0
            test_loss.append(loss.item())
            mask.append(target.cpu().detach().numpy())
            pred.append(out.cpu().detach().numpy())
        avg_test_loss = sum(test_loss) / len(test_loss)
        print('Test loss = {:.{prec}f}'.format(avg_test_loss, prec=4))
    return mask,pred


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimal_model = smp.DeepLabV3Plus(encoder_name='resnet101').to(device)
checkpoint = torch.load(a1_path)
optimal_model.load_state_dict(checkpoint['model_dict'])
optimal_model.eval()
bce_loss = nn.BCEWithLogitsLoss()

mask,pred = test(optimal_model,test_loader)
gt_map = np.concatenate(mask, axis=0)
pred_map = np.concatenate(pred, axis=0)


# In[ ]:


pred_map_sig = nn.Sigmoid()(torch.tensor(pred_map))
threshold = 0.5
predict = deepcopy(pred_map_sig)
predict[pred_map_sig>=threshold]=1
predict[pred_map_sig<threshold]=0

cut_pred1 = []
cut_gt = []
for i in range(predict.shape[0]):
    cut_pred1.append(np.uint8(predict[i,0,132:252,180:300]))
    cut_gt.append(np.uint8(gt_map[i,0,132:252,180:300]))

metrics(cut_gt,cut_pred1)




