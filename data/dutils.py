import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.misc

def load_hgg_lgg_files(path):
    
    return glob.glob(path+'/HGG/*') + glob.glob(path+'/LGG/*')

def load_val_file(path):
    return glob.glob(path+'/*')

def load_nii_to_array(path):
    
    image = nib.load(path)
    image = image.get_data()

    image = np.transpose(image, [2, 1, 0])
    
    return image

def make_image_label(path):
   
    pathes = glob.glob(path+'/*.nii.gz')
    image = []
    seg = None

    for p in pathes:
        if 'flair.nii' in p:
            flair = load_nii_to_array(p)
        elif 't2.nii' in p:
            t2 = load_nii_to_array(p)
        elif 't1.nii' in p:
            t1 = load_nii_to_array(p)
        elif 't1ce.nii' in p:
            t1ce = load_nii_to_array(p)
        else:
            seg = load_nii_to_array(p)
    image.append(flair)
    image.append(t1)
    image.append(t1ce)
    image.append(t2)

    label = seg
    return image, label

def get_box(image, margin):
    
    shape = image.shape
    nonindex = np.nonzero(image) 

    margin = [margin] * len(shape)

    index_min = []
    index_max = []

    for i in range(len(shape)):
        index_min.append(nonindex[i].min())
        index_max.append(nonindex[i].max())
    
    
    for i in range(len(shape)):
        index_min[i] = max(index_min[i] - margin[i], 0)
        index_max[i] = min(index_max[i] + margin[i], shape[i]-1)
    
    # print(index_min)
    # print(index_max)
    return index_min, index_max

def make_box(image, index_min, index_max, data_box):
   
    shape = image.shape

    for i in range(len(shape)):

      
        mid = (index_min[i] + index_max[i])/2
        index_min[i] = mid - data_box[i]/2
        index_max[i] = mid + data_box[i]/2
        
        flag = index_max[i] - shape[i]
        if flag > 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag
        
        flag = index_min[i]
        if flag < 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag
        
        # print('index[%s]: '%i, index_min[i], index_max[i])

        if index_max[i] - index_min[i] != data_box[i]:
            index_max[i] = index_min[i] + data_box[i]
    
        index_max[i] = int(index_max[i])
        index_min[i] = int(index_min[i])

        # print('after index[%s]: '%i, index_min[i], index_max[i])
    return index_min, index_max
    
def crop_with_box(image, index_min, index_max):
    
    # return image[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]
    x = index_max[0] - index_min[0] - image.shape[0]
    y = index_max[1] - index_min[1] - image.shape[1]
    z = index_max[2] - index_min[2] - image.shape[2]
    img = image
    img1 = image
    img2 = image

    if x > 0:
        img = np.zeros((image.shape[0]+x, image.shape[1], image.shape[2]))
        img[x//2:image.shape[0]+x//2, :, :] = image[:, :, :]
        img1 = img

    if y > 0:
        img = np.zeros((img1.shape[0], img1.shape[1]+y, img1.shape[2]))
        img[:, y//2:image.shape[1]+y//2, :] = img1[:, :, :]
        img2 = img

    if z > 0:
        img = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]+z))
        img[:, :, z//2:image.shape[2]+z//2] = img2[:, :, :]

    return img[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]


def normalization(image):
   
    img = image[image>0]
    image = (image - img.mean()) / img.std()
    return image

def get_WT_labels(image):
    return (image == 1) * 1.0 + (image == 2) * 1.0 + (image == 4) * 1.0

def get_TC_labels(image):
    return (image == 1) * 1.0 + (image == 4) * 1.0

def get_ET_labels(image):
    return (image == 4) * 1.0

def get_NCR_NET_label(image):
    """
    ET: enhancing tumor
    For ET task.
    :param image:
    :return:
    """
    return (image == 1)*1.0

def get_precise_labels(image):
    return image*1.0


def load_image_path(path):
    return glob.glob(path+'/1/*') + glob.glob(path+'/2/*') + glob.glob(path+'/3/*')
