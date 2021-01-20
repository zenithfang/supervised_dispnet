from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize
#import pdb
#from skimage.transform import rescale
from scipy.ndimage.interpolation import zoom
#import cv2 as cv
'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, gt_depth, intrinsics):
        #for gt_scale
        for t in self.transforms:
            images, gt_depth, intrinsics = t(images, gt_depth, intrinsics)
        return images, gt_depth, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, gt_depth, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        #check when gt_depth is turned into tensor
        #print(type(gt_depth))
        return images, gt_depth, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, gt_depth, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)#;pdb.set_trace()#this /255 is the difference between the nyu depth implementation and kitti
            #print(type(gt_depth))
            gt_depth_tensor=torch.from_numpy(gt_depth).float()
        return tensors, gt_depth_tensor, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, gt_depth, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            output_depth = np.copy(np.fliplr(gt_depth))
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
            output_depth = gt_depth
        #print(type(output_depth))
        return output_images, output_depth, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, gt_depth, intrinsics, zoom=False):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        #scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]
        #data augment for ground truth depth
        if zoom:
            scaled_gt = zoom(gt_depth, (scaled_h/in_h, scaled_w/in_w))
        else:
            gt_scale = np.amax(gt_depth)
            scaled_gt = imresize(gt_depth, (scaled_h, scaled_w))/255.0*gt_scale#test
            #scaled_gt = imresize(gt_depth, (scaled_h, scaled_w),interp='bilinear')/255.0*gt_scale#bilinear performs badly
        # print("gt_scale{}".format(gt_scale))
        # print("gt_depth")
        # print(gt_depth)
        # print("scaled_gt")
        # print(scaled_gt);# pdb.set_trace()

        #print(gt_depth); pdb.set_trace()
        #scaled_gt = zoom(gt_depth, (y_scaling, x_scaling))/255.0*gt_scale#test

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
        #data augment for ground truth depth
        cropped_gt = scaled_gt[offset_y:offset_y + in_h, offset_x:offset_x + in_w]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y
        #print(type(cropped_gt))
        return cropped_images, cropped_gt, output_intrinsics
