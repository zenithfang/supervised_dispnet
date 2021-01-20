from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from inverse_warp import inverse_warp
import pdb
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
#add supervised loss



def DORN_loss(gt_depth, ord_labels, target,datasets):
    """
    :param ord_labels: ordinal labels for each position of Image I.
    :param target:     the ground_truth discreted using SID strategy.
    :return: ordinal loss
    """

    loss = 0.0
    N, C, H, W = ord_labels.size()
    ord_num = C

    #add in valid for sparse supervision
    if datasets == 'kitti':
        valid = torch.unsqueeze((gt_depth > 0) & (gt_depth < 80), 1)#;pdb.set_trace() #4*1*128*416
    elif datasets == 'nyu':
        valid = torch.unsqueeze((gt_depth > 0) & (gt_depth < 10), 1)#;pdb.set_trace() #4*1*128*416
    else:
        raise "undefined datasets"

    fit_valid = valid.repeat(1, ord_num, 1, 1)

    loss = 0.0

    if torch.cuda.is_available():
        K = torch.zeros((N, C, H, W), dtype=torch.int).cuda()
        for i in range(ord_num):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).cuda()#;pdb.set_trace()
    else:
        K = torch.zeros((N, C, H, W), dtype=torch.int)
        for i in range(ord_num):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int)
    # for comparison
    target_m = torch.unsqueeze(target, 1)#;pdb.set_trace()#4*1*128*416

    # mask_0 = (K <= target_m).detach()
    # mask_1 = (K > target_m).detach()
    # one = torch.ones(ord_labels[mask_1].size())
    # if torch.cuda.is_available():
    #     one = one.cuda()

    # self.loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
    #              + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

    mask_0 = (K <= target_m-1).detach()#according to paper, this should be k < target_m or k <= target_m -1 
    mask_1 = (K > target_m-1).detach()

    one = torch.ones(ord_labels[mask_1 & fit_valid].size())
    if torch.cuda.is_available():
        one = one.cuda()

    loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0 & fit_valid], min=1e-8, max=1e8))) \
                 + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1 & fit_valid], min=1e-8, max=1e8)))

    # N = N * H * W
    # self.loss /= (-N)  # negative

    num_valid = valid.sum().to(torch.float32)
    loss /= (-num_valid)
    return loss


def l2_loss(gt_depth,depth,datasets):
    
    pred_depth =depth[0][:,0] #same tensor shape 4*128*416 as gt_depth(only the unscaled scale)
    loss = 0

    if datasets == 'kitti':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 80)        
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 80)#;pdb.set_trace()
            #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
            loss += ((valid_gt-valid_pred)**2).mean()
    elif datasets == 'nyu':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 10)
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 10)#; pdb.set_trace()
            #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
            loss += (valid_gt-valid_pred).abs().mean()
    else:
        raise "undefined datasets"

    loss = loss/pred_depth.size()[0] #batch size equal 4
    return loss

def l1_loss(gt_depth,depth,datasets):
    
    pred_depth =depth[0][:,0]#;pdb.set_trace() #kitti: same tensor shape 4*128*416 as gt_depth(only the unscaled scale)
    loss = 0

    if datasets == 'kitti':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 80)      
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 80)#; pdb.set_trace()
            #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
            loss += (valid_gt-valid_pred).abs().mean()
    elif datasets == 'nyu':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 10)
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 10)#;pdb.set_trace()
            #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
            loss += (valid_gt-valid_pred).abs().mean()
    else:
        raise "undefined datasets"

    loss = loss/pred_depth.size()[0] #batch size equal 4
    return loss
    
def berhu_loss(gt_depth,depth,datasets):
    pred_depth =depth[0][:,0] #same tensor shape 4*128*416 as gt_depth(only the unscaled scale)
    loss = 0

    if datasets == 'kitti':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 80)        
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 80);# pdb.set_trace()

            residual = (valid_gt-valid_pred).abs()
            max_res = residual.max()
            condition = 0.2*max_res
            L2_loss = ((residual**2+condition**2)/(2*condition))
            L1_loss = residual
            loss += torch.where(residual > condition, L2_loss, L1_loss).mean()
    elif datasets == 'nyu':
            valid = (current_gt > 0) & (current_gt < 10)        
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 10);# pdb.set_trace()

            residual = (valid_gt-valid_pred).abs()
            max_res = residual.max()
            condition = 0.2*max_res
            L2_loss = ((residual**2+condition**2)/(2*condition))
            L1_loss = residual
            loss += torch.where(residual > condition, L2_loss, L1_loss).mean()
    loss = loss/pred_depth.size()[0] #batch size equal 4    
    return loss 

def Scale_invariant_loss(gt_depth,depth,datasets):
    pred_depth =depth[0][:,0] #same tensor shape 4*128*416 as gt_depth(only the unscaled scale)
    loss = 0

    if datasets == 'kitti':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 80)        
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 80);# pdb.set_trace()
            num_valid = valid.sum().to(torch.float32)
            #scalar = torch.cuda.tensor(0.5)/(num_valid**2)
            #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
            loss += ((valid_gt.abs()-valid_pred.abs())**2).mean()-torch.mul((valid_gt-valid_pred).sum()**2,0.5)/(num_valid**2)
    elif datasets == 'nyu':
        for current_gt, current_pred in zip(gt_depth, pred_depth):
            valid = (current_gt > 0) & (current_gt < 10)        
            #valid = valid & crop_mask               
            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, 10);# pdb.set_trace()
            num_valid = valid.sum().to(torch.float32)
            #scalar = torch.cuda.tensor(0.5)/(num_valid**2)
            #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
            loss += ((valid_gt.abs()-valid_pred.abs())**2).mean()-torch.mul((valid_gt-valid_pred).sum()**2,0.5)/(num_valid**2)

    loss = loss/(pred_depth.size()[0])#.to(torch.float32) #batch size equal 4
    return loss

def generate_max_pyramid(image):
    # TODO resize area
    pyramid = [image]
    for i in range(3):
        pyramid.append(F.max_pool2d(pyramid[i], 2, 2))
    return pyramid

def generate_avg_pyramid(image):
    # TODO resize area
    pyramid = [image]
    for i in range(3):
        pyramid.append(F.avg_pool2d(pyramid[i], 2, 2))
    return pyramid

def generate_bilinear_pyramid(image):
    # TODO resize area
    pyramid = [image]

    unsqu_image = torch.unsqueeze(image,1)
    unsqu_pyramid = [unsqu_image]

    for i in range(3):
        unsqu_pyramid.append(F.interpolate(unsqu_pyramid[i], scale_factor=0.5, mode='bilinear', align_corners=False))
        pyramid.append(torch.squeeze(unsqu_pyramid[i+1]))
    return pyramid

def Multiscale_L1_loss(gt_depth,depth,pool_type="bilinear"):
    pred_depth_list=[]
    for i in range(len(depth)):
        pred_depth_list.append(torch.squeeze(depth[i]))
    if pool_type=="max":
        gt_depth_list=generate_max_pyramid(gt_depth)
    elif pool_type=="avg":
        gt_depth_list=generate_avg_pyramid(gt_depth)
    elif pool_type=="bilinear":
        gt_depth_list=generate_bilinear_pyramid(gt_depth)
    else:
        raise "undefined pool type"
    loss = 0
    for i in range(len(depth)):
        current_gt, current_pred = gt_depth_list[i], pred_depth_list[i]
        valid = (current_gt > 0) & (current_gt < 80)        
        #valid = valid & crop_mask               
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80);# pdb.set_trace()
        #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
        loss += (valid_gt-valid_pred).abs().mean()/(2**i)#smoothness scale variant can be other(since the smoothness one is also different)
    return loss

def Multiscale_FULL_L1_loss(gt_depth,depth,pool_type="bilinear"):
    pred_depth_list=[]
    gt_depth_list=[]
    for i in range(len(depth)):
        enlarged_depth = F.upsample(depth[i], scale_factor=2**i, mode=pool_type)
        pred_depth_list.append(torch.squeeze(enlarged_depth))
        gt_depth_list.append(gt_depth)#4*128*416

    loss = 0
    for i in range(len(depth)):
        current_gt, current_pred = gt_depth_list[i], pred_depth_list[i]
        valid = (current_gt > 0) & (current_gt < 80)        
        #valid = valid & crop_mask               
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)#;pdb.set_trace()
        #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
        loss += (valid_gt-valid_pred).abs().mean()/(2**i)#smoothness scale variant can be other(since the smoothness one is also different)
    return loss

def Multiscale_L2_loss(gt_depth,depth):
    pred_depth_list=[]
    for i in range(len(depth)):
        pred_depth_list.append(torch.squeeze(depth[i]))
    gt_depth_list=generate_bilinear_pyramid(gt_depth)#;pdb.set_trace()
    loss = 0
    for i in range(len(depth)):
        current_gt, current_pred = gt_depth_list[i], pred_depth_list[i]
        valid = (current_gt > 0) & (current_gt < 80)        
        #valid = valid & crop_mask               
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80);# pdb.set_trace()
        #loss += ((valid_gt.to(torch.float32).abs()-valid_pred.abs())**2).mean()
        loss += ((valid_gt-valid_pred)**2).mean()/(2**i)#smoothness scale variant can be other(since the smoothness one is also different)
    return loss
     
def Multiscale_berhu_loss(gt_depth,depth):
    pred_depth_list=[]
    for i in range(len(depth)):
        pred_depth_list.append(torch.squeeze(depth[i]))
    gt_depth_list=generate_bilinear_pyramid(gt_depth)#;pdb.set_trace()

    loss = 0
    for i in range(len(depth)):
        current_gt, current_pred = gt_depth_list[i], pred_depth_list[i]
        valid = (current_gt > 0) & (current_gt < 80)        
        #valid = valid & crop_mask               
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80);# pdb.set_trace()

        residual = (valid_gt-valid_pred).abs()
        max_res = residual.max()
        condition = 0.2*max_res
        L2_loss = ((residual**2+condition**2)/(2*condition))
        L1_loss = residual
        loss += torch.where(residual > condition, L2_loss, L1_loss).mean()/(2**i)
        
    return loss 

def Multiscale_scale_inv_loss(gt_depth,depth):
    pred_depth_list=[]
    for i in range(len(depth)):
        pred_depth_list.append(torch.squeeze(depth[i]))
    gt_depth_list=generate_bilinear_pyramid(gt_depth)#;pdb.set_trace()

    loss = 0
    for i in range(len(depth)):
        current_gt, current_pred = gt_depth_list[i], pred_depth_list[i]
        valid = (current_gt > 0) & (current_gt < 80)        
        #valid = valid & crop_mask               
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80);# pdb.set_trace()
        num_valid = valid.sum().to(torch.float32)

        scale_loss = ((valid_gt.abs()-valid_pred.abs())**2).mean()-torch.mul((valid_gt-valid_pred).sum()**2,0.5)/(num_valid**2)
        loss += scale_loss/(2**i)
    return loss

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * out_of_bound

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss += one_scale(d, mask)
    return loss


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        #scaled_map=scaled_map.clamp(1e-3,80)
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

def smooth_DORN_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss = dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()

    return loss

@torch.no_grad()
def compute_errors(gt, pred, dataset='kitti',crop=True,unsupervised=False):
    abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0,0,0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset=='kitti':
        if crop:
            crop_mask = gt[0] != gt[0]
            y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
            x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
            crop_mask[y1:y2,x1:x2] = 1
            max_depth = 80
    else: # I suppose that the nyu depth do not have this crop
        crop_mask = gt[0] ==gt[0]
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < max_depth)

        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)
        
        # for unsupervised training, this median compensation is needed
        if unsupervised:
            valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        
        rmse += torch.sqrt(torch.mean((valid_gt - valid_pred)**2))
        rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) - torch.log(valid_pred)) ** 2))
    
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]]
