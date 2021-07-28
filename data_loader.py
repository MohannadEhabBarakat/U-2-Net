# data loader
from __future__ import print_function, division
import glob
import torch
# from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import cv2

#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


def get_image_with_padding_square_normalized(path,target_shape):
    """
    takes an images and pads it in center, and returns the padding along each side
    Algorithm 
    1. pad to square image
    2. resize
    input_image : 
    target_shape : tuple of the target shape (H,W)
    returns : padded image, tuple with four numbers (left padding,top,right,down)
    """
    # if decode: path = path.decode("utf-8") 
    input_image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    # print("input_image.shape: ", input_image.shape)
    input_h,input_w = input_image.shape[:2]
    max_dim = max(input_h,input_w)
    if max_dim % 2 != 0:
        max_dim += 1
    if len(input_image.shape) == 3:
        channels = 3
        padded_shape = (max_dim,max_dim,input_image.shape[-1])
    elif len(input_image.shape) == 2:
        channels = 1
        padded_shape = (max_dim,max_dim)
    else:
        raise ValueError("Unsupported image dimensions. Expected 2 or 3 but got {0} dims"\
                        .format(len(len(input_image.shape))))
    # print("channels: ", channels)
    padded_image = np.zeros(shape=padded_shape,dtype=input_image.dtype)
    left_pad = (max_dim - input_w) // 2
    right_pad = np.ceil((max_dim - input_w) / 2.0)
    top_pad = (max_dim - input_h) // 2
    bottom_pad = np.ceil((max_dim - input_h) / 2.0)
    padded_image[top_pad:input_h+top_pad,left_pad:input_w+left_pad] = input_image
    img = cv2.resize(padded_image,tuple(target_shape),cv2.INTER_CUBIC)
    cv2.imwrite('temp/contours_d.png',img) 

    # due to opencv resizing
    
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=0)

    # print("img ..")
    # print(img.max(), img.min())
    # print("===========")
    # print("img...")
    img = np.asarray(img,np.float32)
	
    # print("channels ", channels)
    for i in range(channels):
        # print("img.shape", img.shape)
        min_ = img[i,:,:].min()
        max_ = img[i,:,:].max()
        if max_ == 0:
            continue
        img[i,:,:] = (img[i,:,:]-min_)/(max_-min_)  
    # print("img 2..")
    # print(img.max(), img.min())
    # print("&&&&&&&&&&&")
    return img

def get_mask_with_padding_square(path,target_shape):
    """
    takes an images and pads it in center, and returns the padding along each side
    Algorithm 
    1. pad to square image
    2. resize
    input_image : 
    target_shape : tuple of the target shape (H,W)
    returns : padded image, tuple with four numbers (left padding,top,right,down)
    """
    input_image = cv2.imread(path,cv2.IMREAD_UNCHANGED)

    input_image[input_image >= 5] = 255
    input_image[input_image < 5] = 0
    
    input_h,input_w = input_image.shape[:2]
    max_dim = max(input_h,input_w)
    if max_dim % 2 != 0:
        max_dim += 1
    if len(input_image.shape) == 3:
        padded_shape = (max_dim,max_dim,input_image.shape[-1])
    elif len(input_image.shape) == 2:
        padded_shape = (max_dim,max_dim)
    else:
        raise ValueError("Unsupported image dimensions. Expected 2 or 3 but got {0} dims"\
                        .format(len(len(input_image.shape))))
    padded_image = np.zeros(shape=padded_shape,dtype=input_image.dtype)
    left_pad = (max_dim - input_w) // 2
    right_pad = np.ceil((max_dim - input_w) / 2.0)
    top_pad = (max_dim - input_h) // 2
    bottom_pad = np.ceil((max_dim - input_h) / 2.0)
    padded_image[top_pad:input_h+top_pad,left_pad:input_w+left_pad] = input_image
    
    img = cv2.resize(padded_image,tuple(target_shape),cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img[img < 128] = 0.
    img[img >= 128] = 1.
    
    return np.expand_dims(img, 0)

def load_image(src, x, y):
    '''
    load image
    Arguments:
        src      -- dataset source / split name
        x -- name of image
        y -- name of mask

    Returns:
        (image, mask)
    '''
    images_src = "./"+src+"/"+"images"
    masks_src = "./"+src+"/"+"masks"
    
    x = images_src+"/"+x
    y = masks_src+"/"+y

    target_shape = (256, 256)
    x = get_image_with_padding_square_normalized(x, target_shape)
    y = get_mask_with_padding_square(y, target_shape)

    return x, y

class DatasetPipline(Dataset):
	def __init__(self, src, split, is_augmented=True):
		is_augmented = "Augmented" if is_augmented else "Original"
		self.src = os.path.join(src, "acdc", is_augmented, split)
		images_src = self.src+"/"+"images"
		masks_src = self.src+"/"+"masks"
		self.images = sorted(os.listdir(images_src))
		self.masks = sorted(os.listdir(masks_src))

	def __len__(self):
		return len(self.images)

	def __getitem__(self,idx):

		x, y = self.images[idx], self.masks[idx]
		x, y = load_image(self.src, x, y)

		# print("x, ..")
		# print(x.max(), x.min())
		# print("************")

		# x, y = np.expand_dims(x, -1), np.expand_dims(y, -1)

		# print(np.shape(x))

		return {'image':x, 'label':y}


if __name__ == "__main__":

	print(DatasetPipline(src="data", split="Train")[0]["image"])