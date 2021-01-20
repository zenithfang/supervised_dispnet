import os
import numpy as np
from skimage.transform import warp, AffineTransform
import pdb
import torch
import torch.utils.data as data
import torchvision.utils
from torchvision.transforms import Lambda, Normalize, ToTensor
from .image_utils import (EnhancedCompose, Merge, RandomCropNumpy, Split, to_tensor,
                          BilinearResize, CenterCropNumpy, RandomRotate, AddGaussianNoise,
                          RandomFlipHorizontal, RandomColor)
#imagenet pretrained normalization
NYUD_MEAN = [0.485, 0.456, 0.406]
NYUD_STD = [0.229, 0.224, 0.225]
# NYUD_MEAN = [0.48056951, 0.41091299, 0.39225179]
# NYUD_STD = [0.28918225, 0.29590312, 0.3093034]



def transform_chw(transform, lst):
    """Convert each array in lst from CHW to HWC"""
    return transform([x.transpose((1, 2, 0)) for x in lst])
    #return transform(lst.transpose((1, 2, 0)))

class NYU_Depth_V2(data.Dataset):
    def __init__(self, root, split='test', transform=None, limit=None, debug=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.limit = limit
        self.debug = debug

        if debug:
            self.images = np.random.rand(20, 3, 240, 320) * 255
            self.depths = np.random.rand(20, 1, 240, 320) * 10
        elif split == 'test':
            folder = os.path.join(root, 'nyu_depth_v2_other_resolution', 'labeled', 'npy')
            self.images = np.load(os.path.join(folder, 'images.npy'))
            self.depths = np.load(os.path.join(folder, 'depths.npy'))
        else:
            folder = os.path.join(root, 'nyu_depth_v2_other_resolution', 'npy')
            self.file_paths = [os.path.join(folder, n) for n in sorted(os.listdir(folder))]

    def __len__(self):
        if hasattr(self, 'images'):
            length = len(self.images)
        else:
            length = len(self.file_paths)
        if self.limit is not None:
            length = np.minimum(self.limit, length)
        return length

    def __getitem__(self, index):
        if self.split == 'test' or self.debug:
            image = self.images[index]
            depth = self.depths[index]
        else:
            stacked = np.load(self.file_paths[index])
            image = stacked[0:3]
            depth = stacked[3:5]

        if self.transform is not None:
            # image and depth is of datatype float64 and resolution by(292,384)
            #image, depth = image.astype(np.float32), depth.astype(np.float32) there are operation in transform that make the image and depth into double
            image, depth = transform_chw(self.transform, [image, depth])
        # if want to be incorporated in unsupervised setting, it need to keep shape as tgt_img, ref_imgs, intrinsics, intrinsics_inv, gt_depth
        return image, depth

    def compute_image_mean(self):
        return np.mean(self.images / 255, axis=(0, 2, 3))

    def compute_image_std(self):
        return np.std(self.images / 255, axis=(0, 2, 3))

    @staticmethod
    def get_transform(training=True, size=(256, 352), normalize=True):#(284, 392) is the resolution of input 
                                                                      #(257, 353) same as DORN, we choose (352, 256) due to 32 times of dispnet
        if training:
            transforms = [
                Merge(),
                RandomFlipHorizontal(),
                RandomRotate(angle_range=(-5, 5), mode='constant'),
                #crop size (257, 353) is same as the DORN one and this is not suitable for dispnet since it is not the int times of 32
                RandomCropNumpy(size=size),
                RandomAffineZoom(scale_range=(1.0, 1.5)),
                Split([0, 3], [3, 5]), #split merged data into rgb and depth
                # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
                [RandomColor(multiplier_range=(0.8, 1.2)), None],
            ]
        else:
            transforms = [
                # depth>0 can crop the edge of test image 
                [BilinearResize(320/480,448/640), None],# for the most similar scale difference between train and raw dataset
            ]
            
            # transforms = [
            #     #[CenterCropNumpy(size=size),CenterCropNumpy(size=size)]
            #     [BilinearResize(320/480,448/640), None],# for the most similar scale difference between train and raw dataset
            # ]


        transforms.extend([
            # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not

            [ToTensor(), Lambda(to_tensor)],#this ToTensor did not maps from [0, 255] to [0, 1]
             Double_Float(),
            [Normalize(mean=NYUD_MEAN, std=NYUD_STD), None] if normalize else None
        ])

        return EnhancedCompose(transforms)


class RandomAffineZoom():
    def __init__(self, scale_range=(1.0, 1.5), random_state=np.random):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range
        self.random_state = random_state

    def __call__(self, image):
        scale = self.random_state.uniform(self.scale_range[0],
                                          self.scale_range[1])
        if isinstance(image, np.ndarray):
            af = AffineTransform(scale=(scale, scale))
            image = warp(image, af.inverse)
            rgb = image[:, :, 0:3]
            depth = image[:, :, 3:4] / scale
            mask = image[:, :, 4:5]
            return np.concatenate([rgb, depth, mask], axis=2)
        else:
            raise Exception('unsupported type')

class Double_Float():
    def __init__(self):
        pass
    def __call__(self, image):
        #print(image[0].size())#torch.Size([3, 208, 256])
        #print(image[0].type())#double tensor
        #print(len(image))#;pdb.set_trace()
        return [image[0].float(), image[1].float()]
