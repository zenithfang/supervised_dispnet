from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import os
import pdb

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['photo_loss_weight'] = 'p'
    keys_with_prefix['mask_loss_weight'] = 'm'
    keys_with_prefix['smooth_loss_weight'] = 's'
    keys_with_prefix['network'] = 'network'
    keys_with_prefix['pretrained_encoder'] = 'pretrained_encoder'
    keys_with_prefix['loss'] = 'loss'
    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    #for store_true option to be written into the folder name(added here)
    # if args.pretrained_encoder:
    #     folder_string.append('pretrained_encoder')

    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow', channel_first=True):
    tensor = tensor.detach().cpu() #;pdb.set_trace()
    if max_value is None:
        max_value = tensor.max().item()

    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
        if channel_first:
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        if not channel_first:
            array = array.transpose(1, 2, 0)
    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, epoch, filename='checkpoint.pth.tar',record=False):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))
    
    if record:
        #timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        record_path = save_path/"weights_{}".format(epoch)
        record_path.makedirs_p()
        torch.save(dispnet_state, record_path/'dispnet_{}'.format(filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))

# def get_depth_sid(args, labels):
#     if args.dataset == 'kitti':
#         min = 0.001
#         max = 80.0
#         K = 71.0
#     elif args.dataset == 'nyu':
#         min = 0.02
#         max = 80.0
#         K = 68.0
#     else:
#         print('No Dataset named as ', args.dataset)
def get_depth_sid(labels, ordinal_c=71.0,dataset='kitti'):
    # min = 0.001
    # max = 80.0

    # set as consistant with paper to add min value to 1 and set min as 0.01 (cannot converge on both nets)
    if dataset == 'kitti':
        alpha_ = 1.0
        beta_ = 80.999
    elif dataset == 'nyu' or dataset == 'NYU':# for the args in test_disp is different from train
        alpha_ = 1.0
        beta_ = 10.999

    K = float(ordinal_c)#;pdb.set_trace()

    if torch.cuda.is_available():
        alpha_ = torch.tensor(alpha_).cuda()
        beta_ = torch.tensor(beta_).cuda()
        K_ = torch.tensor(K).cuda()
        #;pdb.set_trace()
    else:
        alpha_ = torch.tensor(alpha_)
        beta_ = torch.tensor(beta_)
        K_ = torch.tensor(K)

    #depth = alpha_ * (beta_ / alpha_) ** (labels.float() / K_)-0.999
    depth = 0.5*(alpha_ * (beta_ / alpha_) ** (labels.float() / K_)+alpha_ * (beta_ / alpha_) ** ((labels.float()+1.0) / K_))-0.999# for compensation

    return depth.float()


# def get_labels_sid(args, depth):
#     if args.dataset == 'kitti':
#         alpha = 0.001
#         beta = 80.0
#         K = 71.0
#     elif args.dataset == 'nyu':
#         alpha = 0.02
#         beta = 10.0
#         K = 68.0
#     else:
#         print('No Dataset named as ', args.dataset)
def get_labels_sid(depth, ordinal_c=71.0 ,dataset='kitti'):
    #alpha = 0.001
    #beta = 80.0

    # set as consistant with paper to add min value to 1 and set min as 0.01 (cannot converge on both nets)

    if dataset == 'kitti':
        alpha = 1.0
        beta = 80.999#new alpha is 0.01 which is consistant with other network
    elif dataset == 'nyu':
        alpha = 1.0
        beta  = 10.999

    K = float(ordinal_c)

    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    if torch.cuda.is_available():
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()

    # labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    labels = K * torch.log((depth+0.999) / alpha) / torch.log(beta / alpha)
    if torch.cuda.is_available():
        labels = labels.cuda()
    return labels.int()