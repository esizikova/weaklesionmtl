#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from matplotlib import image
from matplotlib.transforms import Bbox
import shutil
from copy import deepcopy
from skimage.measure import label as la
import nibabel as nib
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings("ignore")
from matplotlib.patches import Rectangle


print("Load and save image from .nii file")

path = 'data/'
seg_path = path+'segmentations/'
img_path = path+'volume/'
seg_dir = np.array(os.listdir(seg_path))
img_dir = np.array(os.listdir(img_path))
os.makedirs(path+'new_seg/',exist_ok=True)
os.makedirs(path+'new_img/',exist_ok=True)

for j in range(len(img_dir)):
    mask_=nib.load(seg_path+'segmentation-'+str(j)+'.nii')
    mask=mask_.get_data()
    img_=nib.load(img_path+'volume-'+str(j)+'.nii')
    img = img_.get_data()
    for i in range(mask.shape[2]):
        if (mask[:,:,i]==2).any():
            cv2.imwrite(path+'new_seg/'+'segmentation-'+str(j)+'_'+str(i)+'.png', mask[:,:,i])
            cv2.imwrite(path+'new_img/'+'volume-'+str(j)+'_'+str(i)+'.png', img[:,:,i])


print("Preprocessing")
img_path, gt_path = path+'new_img/',path+'new_seg/'
img_patch_path,gt_patch_path, ell_path,gr_path = path+'newimg_patch/',path+'newgt_patch/', path+'newellipse/',path+'newgrab/'
img_dir, gt_dir = os.listdir(img_path),os.listdir(gt_path)
os.makedirs(img_patch_path,exist_ok=True)
os.makedirs(gt_patch_path,exist_ok=True)
os.makedirs(ell_path,exist_ok=True)
os.makedirs(gr_path,exist_ok=True)
def getRect(mask):
    cand=np.where(mask==1)
    x_min=min(cand[1])
    x_max=max(cand[1])
    y_min=min(cand[0])
    y_max=max(cand[0])
    return np.array([x_min-10,y_min-10,x_max+10,y_max+10])

def get_patch(im, box):
    w = 8
    box1 = np.zeros((4,), dtype=int)
    center = np.round((box[:2] + box[2:]) / 2)
    box1[0] = max(0,box[0]-w)
    box1[1] = max(0,box[1]-w)
    box1[2] = min(box[2]+w,512)
    box1[3] = min(box[3]+w,512)
    w , h = -box1[1]+box1[3]+1,-box1[0]+box1[2]+1
    mg = max(w,h)//2
    box1 = np.zeros((4,), dtype=int)
    box1[0] = np.maximum(0, center[0] - mg)
    box1[1] = np.maximum(0, center[1] - mg)
    box1[2] = np.minimum(im.shape[1] - 1, center[0] + mg - 1)
    box1[3] = np.minimum(im.shape[0] - 1, center[1] + mg - 1)
    patch = im[box1[1]:box1[3] + 1, box1[0]:box1[2] + 1]
    #print(box1)
    return patch.copy() 


def draw_ellipse(gray):
    thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)[1]
    points = np.column_stack(np.where(thresh.transpose() > 0))
    hull = cv2.convexHull(points)
    if hull is None or len(hull)<5:
        return -1,-1,-1,-1,-1
    else:
        ((centx,centy), (width,height), angle) = cv2.fitEllipse(hull)
        #print("center x,y:",centx,centy)
        #print("diameters:",width,height)
        #print("orientation angle:",angle)
        result = gray.copy()
        cv2.ellipse(result, (int(centx),int(centy)), (int(width/2),int(height/2)), angle, 0, 360, (0,0,255), 2)
    return centx,centy,width,height,angle
# Generate ellipse mask
def save_mask(i):
    img = cv2.imread(gt_patch_path+img_patch_dir[i][:-3]+'png')[:,:,0]//255
    #img = np.load(gt_patch_path+img_patch_dir[i][:-3]+'npy')
    sizes = np.shape(img) 
    fake=np.zeros(sizes,np.uint8)
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    centx,centy,width,height,angle = draw_ellipse(255*img.astype('uint8'))
    if centx==-1 and centy==-1:
            shutil.move(gt_patch_path+img_patch_dir[i][:-3]+'png', small_path+'gt/'+img_patch_dir[i][:-3]+'png')
            shutil.move(img_patch_path+img_patch_dir[i][:-3]+'png', small_path+'img/'+img_patch_dir[i][:-3]+'png')
            #shutil.move(gt_patch_path+img_patch_dir[i][:-3]+'npy', small_path+'gt/'+img_patch_dir[i][:-3]+'npy')
            #shutil.move(img_patch_path+img_patch_dir[i][:-3]+'npy', small_path+'img/'+img_patch_dir[i][:-3]+'npy')
            return
    ax.add_patch(Ellipse((centx,centy), height=height, 
                       width=width,
                       angle=angle,
                       edgecolor='white',
                       facecolor='white',
                       linewidth=0))

    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(fake, cmap = 'gray')
    plt.savefig(ell_path+img_patch_dir[i][:-3]+'png', dpi = max(sizes[0],sizes[1]),bbox_inches = 'tight',pad_inches = 0) 
    plt.close()


'''
# visualize the lesion with ground truth mask
i=409
print(img_dir[i])
img = cv2.imread(img_path+img_dir[i])
gt = cv2.imread(gt_path+gt_dir[i])[:,:,0]
gt[gt==1] = 0
gt[gt==2] = 1
plt.imshow(img)
plt.imshow(gt,cmap='jet',alpha=0.5)
plt.gca().add_patch(Rectangle((237,166),63,63,linewidth=1,edgecolor='r',facecolor='none'))
'''

print("Get the cropped patch around each lesion")
for i in range(len(img_dir)):
    #print(i)
    img = cv2.imread(img_path+img_dir[i])
    gt = cv2.imread(gt_path+gt_dir[i])[:,:,0]
    gt[gt==1] = 0
    gt[gt==2] = 1
    new_img = la(gt,background=0)
    for c in range(1,new_img.max()+1):
        single = 1*(new_img==c)
        bbox_ori = getRect(single)
        img_patch = get_patch(img,bbox_ori)
        gt_patch = get_patch(single,bbox_ori)
        cv2.imwrite(img_patch_path+img_dir[i][:-4]+'_'+str(c)+'.png',img_patch)
        cv2.imwrite(gt_patch_path+img_dir[i][:-4]+'_'+str(c)+'.png',255*gt_patch)
        #np.save(img_patch_path+img_dir[i][:-4]+'_'+str(c)+'.npy',img_patch)
        #np.save(gt_patch_path+img_dir[i][:-4]+'_'+str(c)+'.npy',gt_patch)


img_patch_dir =  np.array(os.listdir(img_patch_path))
print("Remove patches with lesion area < 50")
small_path = path+'newsmall/'
os.makedirs(small_path,exist_ok=True)
os.makedirs(small_path+'gt/',exist_ok=True)
os.makedirs(small_path+'img/',exist_ok=True)
threshold = 50
max_area, max_i = 0, 0
for i in range(len(img_patch_dir)):
    #if img_patch_dir[i][-3:]!='npy':
    gt = cv2.imread(gt_patch_path+img_patch_dir[i])
    area = gt[:,:,0].sum()
    if area<threshold:
        print(i)
        shutil.move(gt_patch_path+img_patch_dir[i], small_path+'gt/'+img_patch_dir[i])
        shutil.move(img_patch_path+img_patch_dir[i], small_path+'img/'+img_patch_dir[i])
        #shutil.move(gt_patch_path+img_patch_dir[i][:-3]+'npy', small_path+'gt/'+img_patch_dir[i][:-3]+'npy')
        #shutil.move(img_patch_path+img_patch_dir[i][:-3]+'npy', small_path+'img/'+img_patch_dir[i][:-3]+'npy')
    elif area>max_area:
        max_area, max_i = area,i

print("generate ellipse")
ell_dir = os.listdir(ell_path)
img_patch_dir =  np.array(os.listdir(img_patch_path))
os.makedirs(ell_path,exist_ok=True)
for i in range(len(img_patch_dir)):
    #if img_patch_dir[i][-3:]!='npy':
    if img_patch_dir[i][:-3]+'png' not in ell_dir:
        save_mask(i)

for i in range(len(ell_dir)): 
    im = cv2.imread(ell_path+ell_dir[i])
    im[im>=0.5*255] = 255
    im[im<0.5*255] = 0
    cv2.imwrite(ell_path+ell_dir[i],im)

ellipses = []
ell_dir = sorted(os.listdir(ell_path))
for i in range(len(ell_dir)):
    img = cv2.resize(cv2.imread(ell_path+ell_dir[i])[:,:,0],(120,120))
    ellipses.append(list(draw_ellipse(img.astype('uint8'))))
ellipses = np.stack(ellipses)
np.save(path+'ellipses_total.npy',ellipses)

'''
# visualization of a patch with fitted ellipse and ground truth mask
i,c = 17078,1
file = img_patch_dir[i][:-6]+'_'+str(c)+'.png'
print(file)
plt.imshow(cv2.imread(img_patch_path+file))
plt.imshow(cv2.imread(gt_patch_path+file)[:,:,0],alpha=0.5)
plt.imshow(cv2.imread(ell_path+file)[:,:,0],cmap='jet',alpha=0.5)
'''
print("calculate the iou and dice between ellipse mask and ground truth mask")
iou, dice = [], []
ell_dir = os.listdir(ell_path)
for i in range(len(ell_dir)):
    mas_ = cv2.imread(ell_path+ell_dir[i])
    mas_ = cv2.resize(mas_[:,:,0],(120,120))//255
    pred_ = cv2.resize(cv2.imread(gt_patch_path+ell_dir[i])[:,:,0],(120,120))//255
    inter = np.logical_and(pred_==1,mas_==1)
    union  = np.logical_or(pred_==1,mas_==1)
    iou.append(np.sum(inter)/np.sum(union))
    dice.append(np.sum(pred_[mas_==1])*2.0 / (np.sum(pred_) + np.sum(mas_)))
#print("IOU is",100*np.nanmean(iou))
#print("DICE is",100*np.nanmean(dice))

print("Remove ellipses with iou<0.4")
bad = []
os.makedirs(small_path+'ell/',exist_ok=True)
for i in range(len(iou)):
    if iou[i]!=iou[i] or iou[i]<0.4:
        bad.append(i) 
        shutil.move(gt_patch_path+ell_dir[i], small_path+'gt/'+ell_dir[i])
        shutil.move(img_patch_path+ell_dir[i], small_path+'img/'+ell_dir[i])
        #shutil.move(gt_patch_path+ell_dir[i][:-3]+'npy', small_path+'gt/'+ell_dir[i][:-3]+'npy')
        #shutil.move(img_patch_path+ell_dir[i][:-3]+'npy', small_path+'img/'+ell_dir[i][:-3]+'npy')
        shutil.move(ell_path+ell_dir[i], small_path+'ell/'+ell_dir[i])
#print(bad)

print("Number of removed ellipses:",len(bad))


# ### Generate GrabCut

def getRect(mask):
    cand=np.where(mask==1)
    x_min=min(cand[1])
    x_max=max(cand[1])
    y_min=min(cand[0])
    y_max=max(cand[0])
    w = x_max-x_min
    h = y_max-y_min
    return (x_min+2,y_min+2,w-2,h-2)

def getGrabcut(img,mask):
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    img1 = img*mask[:,:,np.newaxis]
    mask1 = deepcopy(mask)
    
    rect = getRect(mask1)
    cv2.grabCut(img1,mask1,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask1==2)|(mask1==0),0,1).astype('uint8')
    if mask2.sum()<10:
        return mask
    return mask2

img_path = img_patch_path
gt_path = gt_patch_path
el_path = ell_path
im_dir = np.array(sorted(os.listdir(img_path)))
gt_dir = np.array(sorted(os.listdir(gt_path)))
el_dir = np.array(sorted(os.listdir(el_path)))
#im_dir = np.array([item for item in im_dir if item[-3:]!='npy'])
#gt_dir = np.array([item for item in gt_dir if item[-3:]!='npy'])
tr,val,te = np.load('../tr_ind.npy'),np.load('../val_ind.npy'),np.load('../test_ind.npy')
im_train_dir = im_dir[tr]
gt_train_dir = gt_dir[tr]
el_train_dir = el_dir[tr]
im_val_dir = im_dir[val]
gt_val_dir = gt_dir[val]
el_val_dir = el_dir[val]
im_test_dir = im_dir[te]
gt_test_dir = gt_dir[te]
el_test_dir = el_dir[te]

for i in range(len(im_train_dir)):
    img = cv2.resize(cv2.imread(img_path+im_train_dir[i]),(120,120))
    mas = cv2.resize(cv2.imread(el_path+el_train_dir[i]),(120,120))[:,:,0]
    mas[mas>=0.5]=1
    mas[mas<0.5]=0
    a=getGrabcut(img,mas)
    cv2.imwrite(gr_path+el_train_dir[i],255*a)

plt.imshow(cv2.resize(cv2.imread(gt_path+gt_train_dir[177]),(120,120))[:,:,0],cmap='jet')
plt.imshow(cv2.imread(gr_path+el_train_dir[i])[:,:,0],alpha=0.5)
#plt.imshow(img,alpha=0.5)

for i in range(len(im_val_dir)):
    img = cv2.resize(cv2.imread(img_path+im_val_dir[i]),(120,120))
    mas = cv2.resize(cv2.imread(el_path+el_val_dir[i]),(120,120))[:,:,0]
    mas[mas>=0.5]=1
    mas[mas<0.5]=0
    a=getGrabcut(img,mas)
    cv2.imwrite(gr_path+el_val_dir[i],255*a)

for i in range(len(im_test_dir)):
    img = cv2.resize(cv2.imread(img_path+im_test_dir[i]),(120,120))
    mas = cv2.resize(cv2.imread(el_path+el_test_dir[i]),(120,120))[:,:,0]
    mas[mas>=0.5]=1
    mas[mas<0.5]=0
    a=getGrabcut(img,mas)
    cv2.imwrite(gr_path+el_test_dir[i],255*a)


