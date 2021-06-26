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
from torchvision import models
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
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
import random
from random import shuffle
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
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = [os.path.join(img_path, image_id) for image_id in images_dir]
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



img_path = 'data/newimg_patch/'
gt_path = 'data/newgt_patch/'
el_path = 'data/newellipse/'
gr_path = 'data/newgrab/'
im_dir = np.array(sorted(os.listdir(img_path)))
gt_dir = np.array(sorted(os.listdir(gt_path)))
el_dir = np.array(sorted(os.listdir(el_path)))
gr_dir = np.array(sorted(os.listdir(gr_path)))
#im_dir = np.array([item for item in im_dir if item[-3:]!='npy'])
#gt_dir = np.array([item for item in gt_dir if item[-3:]!='npy'])

tr,val,te = np.load('tr_ind.npy'),np.load('val_ind.npy'),np.load('test_ind.npy')
#tr,val,te = list(np.arange(int(len(im_dir)*0.7))), list(np.arange(int(len(im_dir)*0.7), int(len(im_dir)*0.85))), list(np.arange(int(len(im_dir)*0.85),len(im_dir)))
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

model_name = 'checkpoint/LITS_A1.pt'
optimal_model = smp.DeepLabV3Plus(encoder_name='resnet101').to(device)
checkpoint = torch.load(model_name)
optimal_model.load_state_dict(checkpoint['model_dict'])
optimal_model.eval()
bce_loss = nn.BCEWithLogitsLoss()
mask,pred = test(optimal_model,test_loader)
gt_map = np.concatenate(mask, axis=0)
pred_map = np.concatenate(pred, axis=0)
pred_map_sig = nn.Sigmoid()(torch.tensor(pred_map))
threshold = 0.5
predict = deepcopy(pred_map_sig)
predict[pred_map_sig>=threshold]=1
predict[pred_map_sig<threshold]=0
cut_pred = []
cut_gt = []
for i in range(predict.shape[0]):
    cut_pred.append(np.uint8(predict[i,0,132:252,180:300]))
    cut_gt.append(np.uint8(gt_map[i,0,132:252,180:300]))

print("No classification: train with ell",flush=True)
metrics(cut_gt[:-6],cut_pred[:-6])
