#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import cv2
import os
import urllib
from PIL import Image
from matplotlib.patches import Ellipse
from numpy import asarray
from easydict import EasyDict as edict
import yaml
import csv
import zipfile
import sys
import scipy.io as io
from math import degrees, atan2
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation
import nibabel as nib
import math
from copy import deepcopy




os.mkdir('Data_deep')
os.chdir('Data_deep/')  
# # Download data

# URLs for the zip files. Really slow.
links = [
    'https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip',
    'https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip',
    'https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip',
    'https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip',
    'https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip',
    'https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip',
    'https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip',
    'https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip',
    'https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip',
    'https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip',

    'https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip',
    'https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip',
    'https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip',
    'https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip',
    'https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip',
    'https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip',
    'https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip',
    'https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip',
    'https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip',
    'https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip',

    'https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip',
    'https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip',
    'https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip',
    'https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip',
    'https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip',
    'https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip',
    'https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip',
    'https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip',
    'https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip',
    'https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip',

    'https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip',
    'https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip',
    'https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip',
    'https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip',
    'https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip',
    'https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip',
    'https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip',
    'https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip',
    'https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip',
    'https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip',

    'https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip',
    'https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip',
    'https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip',
    'https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip',
    'https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip',
    'https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip',
    'https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip',
    'https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip',
    'https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip',
    'https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip',

    'https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip',
    'https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip',
    'https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip',
    'https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip',
    'https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip',
    'https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip'
]

md5_link = 'https://nihcc.box.com/shared/static/q0f8gy79q2spw96hs6o4jjjfsrg17t55.txt'
urllib.request.urlretrieve(md5_link, "MD5_checksums.txt")  # download the MD5 checksum file
for idx, link in enumerate(links):
    fn = 'Images_png_%02d.zip' % (idx+1)
    print ('downloading', fn, '...')
    urllib.request.urlretrieve(link, fn)  # download the zip file
print ("Download complete. Please check the MD5 checksums")


# # Unzip

for idx in range(0,len(links)):
    with zipfile.ZipFile('Images_png_%02d.zip' % (idx+1),"r") as zip_ref:
        print("unzipping", idx+1)
        zip_ref.extractall()


# ### Download https://nihcc.app.box.com/v/DeepLesion/file/305569842654 manually in the Data_deep directory.

# ## Get the traib, val, test split

infodl = pd.read_csv('DL_info.csv')
train, val, test = [], [], []
for j in range(len(infodl)):
    c = infodl['Train_Val_Test'][j]
    if c == 1:
        train.append(j)
    elif c == 2:
        val.append(j)
    elif c == 3:
        test.append(j)

np.save('train.npy',train)
np.save('val.npy',val)
np.save('test.npy',test)


# ## Preprocessing Functions from Lesanet

# https://github.com/rsummers11/CADLab/blob/master/LesaNet/load_ct_img.py

os.chdir("Images_png")
if sys.version_info[0] >= 3:
    unicode = str

config = edict()
config.BOX_PAD = 60
config.PAD_BORDER = True
# algorithm related params
config.PIXEL_MEANS = np.array([50])
config.NUM_SLICES= 3
config.MAX_IM_SIZE = 512
config.SCALE = 512
config.NORM_SPACING = -1
config.SLICE_INTV = 2
#config.WINDOWING = [-1024, 3071]
config.IMG_DO_CLIP = False  # clip the black borders of ct images
config.ROI_METHOD= 'FIXED_CONTEXT'
config.TRAIN = edict()
config.SAMPLES_PER_BATCH = 256
config.TRAIN.USE_PRETRAINED_MODEL = True
config.TEST = edict()

# default settings
default = edict()

# default network
default.network = 'vgg'
default.base_lr = 0.001

default.dataset = 'DeepLesion'
default.image_set = 'train'

# default training
default.frequent = 20
default.model_path = 'checkpoints/'
default.res_path = 'results/'
default.epoch = 10
default.lr = default.base_lr
default.lr_step = '7'
default.prefetch_thread_num = 4  # 0: no prefetch

default.world_size = 1  # number of distributed processes
default.dist_url = 'tcp://224.66.41.62:23456'  # url used to set up distributed training
default.dist_backend = 'gloo'  # distributed backend
default.seed = None  # seed for initializing training

default.gpus = '0'
default.val_gpu = default.gpus
default.val_image_set = 'val'
default.val_vis = False
default.val_shuffle = False
default.val_max_box = 5
default.val_thresh = 0
default.weight_decay = .0005
default.groundtruth_file = 'DL_info.csv'
default.image_path = ''
default.validate_at_begin = True
default.testing = False

default.flip = False
default.shuffle = True
default.begin_epoch = 0
default.show_avg_loss = 100  # 1: show exact loss of each batch. >1: smooth the shown loss



#### --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes to load and preprocess CT images in DeepLesion.
# --------------------------------------------------------


#from config import config, default


def load_prep_img(imname, slice_idx, spacing, slice_intv, window,do_clip=False, num_slice=3, do_windowing=True):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    if type(imname) in (str, unicode):
        im, mask = load_multislice_img_16bit_png(imname, slice_idx, slice_intv, do_clip, num_slice)
    else:
        im, mask = load_multislice_img_nifti(imname, slice_idx, slice_intv, do_clip, num_slice)

    if do_windowing:
        im = windowing(im, window)

    if do_clip:  # clip black border
        c = get_range(mask, margin=0)
        im = im[c[0]:c[1] + 1, c[2]:c[3] + 1, :]
        # mask = mask[c[0]:c[1] + 1, c[2]:c[3] + 1]
        # print im.shape
    else:
        c = [0, im.shape[0]-1, 0, im.shape[1]-1]

    im_shape = im.shape[0:2]
    if spacing is not None and config.NORM_SPACING > 0:  # spacing adjust, will overwrite simple scaling
        im_scale = float(spacing) / config.NORM_SPACING
    else:
        im_scale = float(config.SCALE) / float(np.min(im_shape))  # simple scaling

    max_shape = np.max(im_shape)*im_scale
    if max_shape > config.MAX_IM_SIZE:
        im_scale1 = float(config.MAX_IM_SIZE) / max_shape
        im_scale *= im_scale1

    if im_scale != 1:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    #im -= config.PIXEL_MEANS
    return im, im_scale, c


def load_multislice_img_16bit_png(imname, slice_idx, slice_intv, do_clip, num_slice):
    data_cache = {}

    def _load_data(imname, delta=0):
        imname1 = get_slice_name(imname, delta)
       # print("load",data_cache.keys())
       # print("load",imname1)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = cv2.imread(fullpath(imname1), -1)
            #plt.imshow(data_cache[imname1]/255.0)
            if data_cache[imname1] is None:
                print ('file reading error:', imname1)
        return data_cache[imname1]

    im_cur = _load_data(imname)
   # print('imcur',im_cur.shape)
    mask = get_mask(im_cur) if do_clip else None

    if config.SLICE_INTV == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice  # only use the central slice

    else:
        ims = [im_cur]
       # print("before if",np.array(ims).shape)
        # find neighboring slices of im_cure
        rel_pos = float(config.SLICE_INTV) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
            for p in range(math.floor((num_slice-1)/2)):
               # print('---',imname)
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range(math.floor((num_slice-1)/2)):
                intv1 = rel_pos*(p+1)
              #  print('---',imname)
                slice1 = _load_data(imname, - np.ceil(intv1))
                slice2 = _load_data(imname, - np.floor(intv1))
                im_prev = a * slice1 + b * slice2  # linear interpolation

                slice1 = _load_data(imname, np.ceil(intv1))
                slice2 = _load_data(imname, np.floor(intv1))
                im_next = a * slice1 + b * slice2
            #    print("im_prev",im_prev.shape,"im_next",im_next.shape)
                ims = [im_prev] + ims + [im_next]

    ims = [im.astype(float) for im in ims]
   # print("ims",np.array(ims).shape)
    im = cv2.merge(ims)
 #   print("merged",im.shape)
    im = im.astype(np.float32, copy=False)-32768
    # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit

    return im, mask


def get_slice_name(imname, delta=0):
    if delta == 0: return imname
    delta = int(delta)
    idx = imname.rindex('_')
    imname = imname[:idx] + os.sep + imname[idx+1:]
    dirname, slicename = imname.split(os.sep)
    slice_idx = int(slicename[:-4])
    #imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
    imname1 = '%s%s%03d.png' % (dirname, '/', slice_idx + delta)
    #print("getname",imname1)
    while not os.path.exists(fullpath(imname1)):  # if the slice is not in the dataset, use its neighboring slice
        delta -= np.sign(delta)
        #imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
        imname1 = '%s%s%03d.png' % (dirname, '/', slice_idx + delta)
        #print ('file not found:', imname1)
        if delta == 0: break

    return imname1


def fullpath(imname):
    imname_full = os.path.join(imname[:-8], imname[-7:])
    #print(imname_full)
    return imname_full


def load_multislice_img_nifti(vol, slice_idx, slice_intv, do_clip, num_slice):
    if do_clip:
        mask = get_mask(vol[:,:,slice_idx])
    else:
        mask = None

    im_cur = vol[:, :, slice_idx]

    if config.SLICE_INTV == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice

    else:
        max_slice = vol.shape[2] - 1
        ims = [im_cur]
        # linear interpolate
        rel_pos = float(config.SLICE_INTV) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:
            for p in range(math.floor((num_slice-1)/2)):
                intv1 = rel_pos*(p+1)
                slice1 = int(max(slice_idx - intv1, 0))
                im_prev = vol[:,:,slice1]
                slice2 = int(min(slice_idx + intv1, max_slice))
                im_next = vol[:,:, slice2]
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range(math.floor((num_slice-1)/2)):
                intv1 = rel_pos*(p+1)
                slice1 = int(max(slice_idx - np.ceil(intv1), 0))
                slice2 = int(max(slice_idx - np.floor(intv1), 0))
                im_prev = a * vol[:,:,slice1] + b * vol[:,:,slice2]

                slice1 = int(min(slice_idx + np.ceil(intv1), max_slice))
                slice2 = int(min(slice_idx + np.floor(intv1), max_slice))
                im_next = a * vol[:, :, slice1] + b * vol[:, :, slice2]

                ims = [im_prev] + ims + [im_next]

    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    return im, mask


def windowing(im, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


# backward windowing
def windowing_rev(im, win):
    im1 = im.astype(float)/255
    im1 *= win[1] - win[0]
    im1 += win[0]
    return im1


def get_mask(im):
    # use a intensity threshold to roughly find the mask of the body
    th = 32000  # an approximate background intensity value
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
    # mask = binary_dilation(mask)
    # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

    if mask.sum() == 0:  # maybe atypical intensity
        mask = im * 0 + 1
    return mask.astype(dtype=np.int32)


def get_range(mask, margin=0):
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return u, d, l, r


def im_list_to_blob(ims, use_max_size=False):
    """Convert a list of images into a network input.
    """
    # max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # min_shape = np.array([im.shape for im in ims]).min(axis=0)
    # print max_shape, min_shape
    if use_max_size:
        max_shape = np.array([config.MAX_IM_SIZE, config.MAX_IM_SIZE])
    else:
        max_shape = np.array([im.shape for im in ims]).max(axis=0)

    num_images = len(ims)
    num_channel = ims[0].shape[2] if ims[0].ndim == 3 else 3
    blob = np.zeros((num_images, num_channel, max_shape[0], max_shape[1]),
                    dtype=np.float32)
    rois = np.zeros((num_images, 4))
    for i in xrange(num_images):
        im = ims[i]

        # # put images in the center
        # m = (max_shape - im.shape) / 2
        # rois[i, :] = np.array([m[1], m[0], m[1] + im.shape[1], m[0] + im.shape[0]])
        # if im.ndim == 2:
        # 	for chn in range(3):
        # 		blob[i, chn, m[0]:m[0] + im.shape[0], m[1]:m[1] + im.shape[1]] = im
        # elif im.ndim == 3:
        # 	blob[i, :, m[0]:m[0] + im.shape[0], m[1]:m[1] + im.shape[1]] = im.transpose((2, 0, 1))

        # put images on the corner
        rois[i, :] = np.array([0, 0, im.shape[1], im.shape[0]])
        if im.ndim == 2:
            for chn in range(num_channel):
                blob[i, chn, :im.shape[0], :im.shape[1]] = im
        elif im.ndim == 3:
            blob[i, :, :im.shape[0], :im.shape[1]] = im.transpose((2, 0, 1))

    return blob, rois


def map_box_back(boxes, cx=0, cy=0, im_scale=1.):
    boxes /= im_scale
    boxes[:, [0,2]] += cx
    boxes[:, [1,3]] += cy
    return boxes


def get_patch(im, box):
    # box = box0.copy()  # shouldn't change box0!
    # if spacing is not None and config.NORM_SPACING > 0:  # spacing adjust, will overwrite simple scaling
    #     im_scale = float(spacing) / config.NORM_SPACING
    #     box *= im_scale

    mg = config.BOX_PAD
    if config.ROI_METHOD == 'FIXED_MARGIN' or config.ROI_METHOD == 'VAR_SIZE_FIXED_MARGIN':
        # method 1: crop real lesion size + margin. will pad zero for diff size patches
        box1 = np.round(box).astype(int)
        box1[0] = np.maximum(0, box1[0] - mg)
        box1[1] = np.maximum(0, box1[1] - mg)
        box1[2] = np.minimum(im.shape[1] - 1, box1[2] + mg)
        box1[3] = np.minimum(im.shape[0] - 1, box1[3] + mg)
        patch = im[box1[1]:box1[3] + 1, box1[0]:box1[2] + 1]

        offset_x = np.maximum(box[0] - mg, 0)
        offset_y = np.maximum(box[1] - mg, 0)
        box_new = box - np.array([offset_x, offset_y] * 2)

        max_shape = np.max(patch.shape)
        patch_scale = 1.
        if max_shape > config.MAX_PATCH_SIZE:
            patch_scale = float(config.MAX_PATCH_SIZE) / max_shape
            patch = cv2.resize(patch, None, None, fx=patch_scale, fy=patch_scale, interpolation=cv2.INTER_LINEAR)
            box_new *= patch_scale

    elif config.ROI_METHOD == 'FIXED_CONTEXT':
        # method 2: crop fixed size context, so no need to pad zeros
        center = np.round((box[:2] + box[2:]) / 2)
        box1 = np.zeros((4,), dtype=int)
        box1[0] = np.maximum(0, center[0] - mg)
        box1[1] = np.maximum(0, center[1] - mg)
        box1[2] = np.minimum(im.shape[1] - 1, center[0] + mg - 1)
        box1[3] = np.minimum(im.shape[0] - 1, center[1] + mg - 1)

        patch = im[box1[1]:box1[3] + 1, box1[0]:box1[2] + 1]

        # handle diff size
        if config.PAD_BORDER:
            xdiff = mg * 2 - patch.shape[1]
            ydiff = mg * 2 - patch.shape[0]
            if xdiff > 0:
                if center[0] - mg < 0:
                    patch = cv2.copyMakeBorder(patch, 0, 0, xdiff, 0, cv2.BORDER_REPLICATE)
                else:
                    patch = cv2.copyMakeBorder(patch, 0, 0, 0, xdiff, cv2.BORDER_REPLICATE)
            if ydiff > 0:
                if center[1] - mg < 0:
                    patch = cv2.copyMakeBorder(patch, ydiff, 0, 0, 0, cv2.BORDER_REPLICATE)
                else:
                    patch = cv2.copyMakeBorder(patch, 0, ydiff, 0, 0, cv2.BORDER_REPLICATE)

        box_new = np.maximum(0, box-np.hstack((center, center))+mg)
        patch_scale = 1.

    return patch.copy(), box_new, patch_scale


def load_DL_info(path):
    # load annotations and meta-info from DL_info.csv
    info = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #filename = row[0]  # replace the last _ in filename with / or \
            #idx = filename.rindex('_')
            #row[0] = filename[:idx] + os.sep + filename[idx+1:]
            #row[0] = os.sep + row[0]
            info.append(row)
    info = info[1:]

    # the information not used in this project are commented
    res = {}
    res['filenames'] = np.array([row[0] for row in info])
    res['patient_idx'] = np.array([int(row[1]) for row in info])
    # info['study_idx'] = np.array([int(row[2]) for row in info])
    # info['series_idx'] = np.array([int(row[3]) for row in info])
    res['slice_idx'] = np.array([int(row[4]) for row in info])
    # res['d_coordinate'] = np.array([[float(x) for x in row[5].split(',')] for row in info])
    # res['d_coordinate'] -= 1
    res['boxes'] = np.array([[float(x) for x in row[6].split(',')] for row in info])
    res['boxes'] -= 1  # coordinates in info file start from 1
    # res['diameter'] = np.array([[float(x) for x in row[7].split(',')] for row in info])
    res['norm_location'] = np.array([[float(x) for x in row[8].split(',')] for row in info])
    res['type'] = np.array([int(row[9]) for row in info])
    res['noisy'] = np.array([int(row[10]) > 0 for row in info])
    # res['slice_range'] = np.array([[int(x) for x in row[11].split(',')] for row in info])
    res['spacing3D'] = np.array([[float(x) for x in row[12].split(',')] for row in info])
    res['spacing'] = res['spacing3D'][:, 0]
    res['slice_intv'] = res['spacing3D'][:, 2]  # slice intervals
    # res['image_size'] = np.array([[int(x) for x in row[13].split(',')] for row in info])
    res['DICOM_window'] = np.array([[float(x) for x in row[14].split(',')] for row in info])
    # res['gender'] = np.array([row[15] for row in info])
    # res['age'] = np.array([float(row[16]) for row in info])  # may be NaN
    res['train_val_test'] = np.array([int(row[17]) for row in info])

    return res




def load_process_img(img_idx):
    imname = str(info['filenames'][img_idx])
    slice_idx = info['slice_idx'][img_idx]
    spacing = info['spacing'][img_idx]
    slice_intv = info['slice_intv'][img_idx]
    box = info['boxes'][img_idx].copy()
    window = info['DICOM_window'][img_idx]
    im, im_scale, crop = load_prep_img(imname, slice_idx, spacing, slice_intv,window,
                                        do_clip=False, num_slice=config.NUM_SLICES)
   # print(imname,slice_idx,spacing,slice_intv,window,im_scale)
    box *= im_scale
    patch, new_box, patch_scale = get_patch(im, box)
    return patch, new_box




info = load_DL_info("../DL_info.csv") 


# # Generate the mat that you could read directly



a={'image':[],'bbox':[],'recist':[]}
raw_images = []
recist_list = []
i=1
n = 10000
for j in range(len(infodl)):
    data_path = 'images_total_'+str(i)+'.mat'
    if len(recist_list)%n==0 and len(recist_list)>0:
        div = len(recist_list)//n
        a['image'] = raw_images[n*(div-1):n*div]
        a['recist'] = recist_list[n*(div-1):n*div]
        io.savemat(data_path,a)
        a={'image':[],'bbox':[],'recist':[]}
        i+=1
    filena = infodl.iloc[j]['File_name'][:-4]
    image, bbox = load_process_img(j)
    recistori = np.array([float(i) for i in infodl.iloc[j]['Measurement_coordinates'].split(',')])
    bboxori = np.array([float(i) for i in infodl.iloc[j]['Bounding_boxes'].split(',')])
    xdist = bbox[0]-bboxori[0]
    ydist = bbox[1]-bboxori[1]
    a['bbox'].append(np.array(bbox))
    recist = []
    for k in range(len(recistori)):
        if k%2==0:
            recist.append(recistori[k]+xdist)
        else:
            recist.append(recistori[k]+ydist)
    recist_list.append(np.array(recist))
    raw_images.append(image)
if len(recist_list)%n>0:
    div = len(recist_list)//n
    a['image'] = raw_images[n*div:]
    a['recist'] = recist_list[n*div:]
    io.savemat(data_path,a)


# ## load data
os.chdir('..')
delete_idx = [24897,3136,5780,9239,14485,29866,14474,6768] 
train_idx = np.load('train.npy')
val_idx = np.load('val.npy')
test_idx = np.load('test.npy')
train_idx = np.setdiff1d(train_idx,delete_idx)
test_idx = np.setdiff1d(test_idx,delete_idx)
val_idx = np.setdiff1d(val_idx,delete_idx)
raw_images=np.array(raw_images)
train_raw = raw_images[train_idx]
val_raw = raw_images[val_idx]
test_raw = raw_images[test_idx]
train_raw = [patch.astype(float) / 255 for patch in train_raw]
test_raw = [patch.astype(float) / 255 for patch in test_raw]
val_raw = [patch.astype(float) / 255 for patch in val_raw]
recist = recist_list


# ## generate ellipses

os.mkdir('mask')
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def dist(x,c):
    x1,y1 = x
    x2,y2 = c
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

# generate and save pseudo mask
def save_mask(index):
    fake=np.zeros((120,120),np.uint8)
    sizes = np.shape(fake)    
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    center = line_intersection(recist[index][:4].reshape(2,2), recist[index][4:].reshape(2,2))
    d1,d2 = dist(recist[index][:2],recist[index][2:4]),dist(recist[index][4:6],recist[index][6:])
    rotation = 90-degrees(atan2(recist[index][6] - recist[index][4], recist[index][7]-recist[index][5]))
    ax.add_patch(Ellipse(center, height=d1, 
                       width=d2,
                       angle=rotation,
                       edgecolor='white',
                       facecolor='white',
                       linewidth=3))

    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(fake, cmap = 'gray')
    plt.savefig('mask/'+str(index)+'.jpg', dpi = sizes[0], bbox_inches = 'tight',pad_inches = 0) 
    plt.close()
    return [center[0],center[1],d2,d1,rotation]

ell_list = []
for i in range(len(raw_images)):
    ell_list.append(save_mask(i))
np.save('ell.npy',np.stack(ell_list))


# ## generate grabcuts

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
    if (mask2==0).all():
        return mask
    return mask2

from matplotlib import image
mask = [image.imread('mask/'+str(i)+'.jpg')[:,:,0] for i in range(len(raw_images))]
mask = np.stack(mask)
m = mask/255.0
mas = np.ma.masked_where((m>0.9),m)
mas = mas.filled(fill_value=1)
mas = np.ma.masked_where((mas!=1),mas)
mas = mas.filled(fill_value=0)
mas = np.uint8(mas)
train_mask=np.array(mas[train_idx])
val_mask=np.array(mas[val_idx])
test_mask=np.array(mas[test_idx])

val_grab=[]
for i in range(len(val_raw)):
    img=np.uint8(val_raw[i])
    mas=val_mask[i]
    newmask = getGrabcut(img,mas)
    val_grab.append(newmask)
np.save('val_grabcut.npy', val_grab)
test_grab=[]
for i in range(len(test_raw)):
    img=np.uint8(test_raw[i])
    mas=test_mask[i]
    newmask = getGrabcut(img,mas)
    test_grab.append(newmask)
np.save('test_grabcut.npy', test_grab)
train_grab=[]
for i in range(len(train_raw)):
    img=np.uint8(train_raw[i])
    mas=train_mask[i]
    newmask = getGrabcut(img,mas)
    train_grab.append(newmask)
np.save('train_grabcut.npy', train_grab)