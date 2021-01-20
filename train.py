import argparse
import time
import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
import models
from datasets.nyu_depth_v2 import NYU_Depth_V2
from utils import tensor2array, save_checkpoint, save_path_formatter
from inverse_warp import inverse_warp
import loss_functions
#from loss_functions import smooth_DORN_loss, DORN_loss, berhu_loss, Multiscale_berhu_loss, l1_loss, Multiscale_L1_loss, Multiscale_L2_loss, l2_loss, Multiscale_scale_inv_loss, Scale_invariant_loss, photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import pdb
import utils
#setup the pretrained model directory
import os
import networks
from layers import disp_to_depth

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--network", default='disp_vgg', type=str, help="network type")
parser.add_argument("--dataset", default='kitti', type=str, help="dataset name")
parser.add_argument('--imagenet-normalization', action='store_true', help='use imagenet parameter for normalization.')
parser.add_argument('--pretrained-encoder', action='store_true', help='use imagenet pretrained parameter.')
parser.add_argument('--loss', default='Multi_L1', type=str, help='loss type')
parser.add_argument('--ordinal-c', default=80, type=int, metavar='N', help='DORN loss channel number')
parser.add_argument('--diff-lr', action='store_true', help='use different learning rate for encoder and decoder')
parser.add_argument('--sgd', action='store_true', help='use sgd optimizer, if not then adam')
parser.add_argument('--record', action='store_true', help='save every epoch checkpoints to check optimizer and loss influence over learning progress')
parser.add_argument("--unsupervised", action='store_true', help="to have unsupervised loss")
parser.add_argument('--data-amount', default=1, type=float, metavar='M', help='percentage of data to be trained')
parser.add_argument("--monodepth2", action='store_true', help="to finetune over monodepth2 model")

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',#change for the real epoch definition that each epoch equals the 
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0)
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')#'store_true' for boolean
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.environ['TORCH_MODEL_ZOO'] = '/scratch_net/minga/zfang/pytorch_init_weight/models'

def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
    save_path = save_path_formatter(args, parser)#;pdb.set_trace()
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.evaluate:
        args.epochs = 0

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code: different dataloader for different dataset
    if args.dataset == 'kitti':
        if args.imagenet_normalization:
            if args.monodepth2:
            # this date normalize of monodepth2 is written inside the dispnet
            # thus we just use this 0 and 1 that do not change data
                normalize = custom_transforms.Normalize(mean=[0, 0, 0],
                                                        std=[1, 1, 1])
            else:
            	normalize = custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
        else:
        	normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5])

        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            #custom_transforms.RandomScaleCrop(),# test without crop
            custom_transforms.ArrayToTensor(),
            normalize
        ])
        # train_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
        #**************************
        valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

        print("=> fetching scenes in '{}'".format(args.data))

    # different dataloader for different dataset
    # add in percentage of data
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            percentage=args.data_amount
        )

        # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
        if args.with_gt:
            from datasets.validation_folders import ValidationSet
            val_set = ValidationSet(
                args.data,
                transform=valid_transform
            )
        else:
            val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
            )

    elif args.dataset == 'nyu':
        train_set = NYU_Depth_V2(
            args.data, 
            split='train', 
            transform=NYU_Depth_V2.get_transform(training=True),
            limit=None, 
            debug=False
        )
        val_set = NYU_Depth_V2(
            args.data, 
            split='test', 
            transform=NYU_Depth_V2.get_transform(training=False),
            limit=None, 
            debug=False
        )

    #nyu loader does not have scene number
    if args.dataset == 'kitti':
        print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
        print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)#;pdb.set_trace()
    # val_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.batch_size, pin_memory=True)#;pdb.set_trace()
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.batch_size, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    #changing different network
    if args.monodepth2:
        #add in code about the monodepth2 model
        # consider about add this part into the monodepth2.py for convenience
        #import networks
        mono2_models = {}
        #optim_params = []
        #configure encoder
        if args.network=='disp_vgg_BN':
            mono2_models["encoder"] = networks.vggEncoder(
                num_layers = 16, pretrained = False).to(device)
        elif args.network=='disp_res_18':    
            mono2_models["encoder"] = networks.ResnetEncoder(
                num_layers = 18, pretrained = False).to(device)
        else:
            raise "undefined network"
        #optim_params += list(mono2_models["encoder"].parameters())
        #configure decoder
        mono2_models["depth"] = networks.DepthDecoder(
            mono2_models["encoder"].num_ch_enc).to(device)
        #optim_params += list(mono2_models["depth"].parameters())
        # when monodepth2, it must load existing weight (not include adam)
        load_model(pretrained_model = mono2_models, weights_folder = args.pretrained_disp)
        #construct this disp_net to be compatiable with existing framework
        disp_net = models.monodepth2(encoder = mono2_models["encoder"], decoder = mono2_models["depth"])
    else:
        if args.network=='dispnet':
        	disp_net = models.DispNetS(datasets=args.dataset).to(device)
        elif args.network=='disp_res':
        	disp_net = models.Disp_res(datasets=args.dataset).to(device)
        elif args.network=='disp_res_50':
            disp_net = models.Disp_res_50(datasets=args.dataset).to(device)
        elif args.network=='disp_res_18':
            disp_net = models.Disp_res_18(datasets=args.dataset).to(device)
        elif args.network=='disp_vgg':
        	disp_net = models.Disp_vgg_feature(datasets=args.dataset).to(device)
        elif args.network=='disp_vgg_BN':
            disp_net = models.Disp_vgg_BN(datasets=args.dataset).to(device)
        elif args.network=='FCRN':
            disp_net = models.FCRN(datasets=args.dataset).to(device)
        elif args.network=='res50_aspp':
            disp_net = models.res50_aspp(datasets=args.dataset).to(device)  
        elif args.network=='ASPP':
            disp_net = models.deeplab_depth(datasets=args.dataset).to(device)  
        elif args.network=='disp_res_101':
            disp_net = models.Disp_res_101(datasets=args.dataset).to(device)
        elif args.network=='DORN':
            disp_net = models.DORN(freeze=args.diff_lr, datasets=args.dataset).to(device)
        elif args.network=='disp_vgg_BN_DORN':
            disp_net = models.Disp_vgg_BN_DORN(ordinal_c=args.ordinal_c, datasets=args.dataset).to(device)
        else:
        	raise "undefined network"

    output_exp = args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=args.mask_loss_weight > 0).to(device)

    if args.pretrained_exp_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_disp and (not args.monodepth2):
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    # for the use of disp_vgg with pretrained

    else:
        if not args.monodepth2: 
            disp_net.init_weights(use_pretrained_weights=args.pretrained_encoder)# decide whether use pretrained encoder
    
    if not args.diff_lr:
        print('=> setting adam solver')

        # optim_params = [
        #     {'params': disp_net.parameters(), 'lr': args.lr},
        #     {'params': pose_exp_net.parameters(), 'lr': args.lr}
        # ]
        #if not args.monodepth2: 
        optim_params = [
            {'params': disp_net.parameters(), 'lr': args.lr}
        ]

        if args.sgd:
            optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(optim_params,
                                         betas=(args.momentum, args.beta),
                                         weight_decay=args.weight_decay)# these 3 parameters are all the default parameter of adam 
    else:
        # set as DORN 
        # different modules have different learning rate
        train_params = [{'params': disp_net.get_1x_lr_params(), 'lr': args.lr},
                        {'params': disp_net.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    if args.pretrained_disp:
        print("=> using pre-trained parameters for adam")
        if not args.monodepth2:#if monodepth2, not load optimizer
            weights = torch.load(args.pretrained_disp)
            optimizer.load_state_dict(weights['optimizer'])

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    if args.pretrained_disp or args.evaluate:
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, 0, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, 0, logger, output_writers)
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, 0)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))#error_names[2:9] errors[2:9] it used to be so
        logger.valid_writer.write(' * Avg {}'.format(error_string))

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()#;pdb.set_trace()
        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, dispnet_state={
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, exp_pose_state={
                'epoch': epoch + 1,
                'state_dict': pose_exp_net.module.state_dict()
            },
            is_best=is_best,
            record=args.record,
            epoch=epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight

    # switch to train mode
    disp_net.train()
    pose_exp_net.train()
    
    # # freeze bn
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    if args.diff_lr:
        disp_net.apply(set_bn_eval)


    end = time.time()
    logger.train_bar.update(0);
    
    # this is for both supervised and unsupervised edition
    # for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, gt_depth) in enumerate(train_loader):
    # this is for supervised only edition
    for i, (tgt_img, gt_depth) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)

        if args.unsupervised:    
            ref_imgs = [img.to(device) for img in ref_imgs]
            intrinsics = intrinsics.to(device)
            intrinsics_inv = intrinsics_inv.to(device); 
            explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)

        gt_depth = gt_depth.to(device)#;pdb.set_trace()
        if args.dataset=='nyu':
            gt_depth = torch.squeeze(gt_depth[:,0,:,:])#;pdb.set_trace()# another data is just mask and this mask is calculated by depth==10 or depth==0


        if args.loss == 'DORN':
            target_c = utils.get_labels_sid(gt_depth, ordinal_c=args.ordinal_c, dataset=args.dataset)
            pred_d, pred_ord = disp_net(tgt_img)
        else:
            disparities = disp_net(tgt_img)
            if args.monodepth2:
                depth = [5.4/disp for disp in disparities]#5.4 is the stereo scale factor 
            else:
                depth = [1/disp for disp in disparities]
            #explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)

        if not args.unsupervised: 
            if args.loss=='Multi_L1':
                loss_1 = loss_functions.Multiscale_L1_loss(gt_depth, depth)
            elif args.loss=='Multi_full_L1':
                loss_1 = loss_functions.Multiscale_FULL_L1_loss(gt_depth, depth)
            elif args.loss=='Multi_berhu':
                loss_1 = loss_functions.Multiscale_berhu_loss(gt_depth, depth)
            elif args.loss=='Multi_L2':
                loss_1 = loss_functions.Multiscale_L2_loss(gt_depth, depth)
            elif args.loss=='L1': 
                loss_1 = loss_functions.l1_loss(gt_depth, depth, args.dataset)
            elif args.loss=='berhu': 
                loss_1 = loss_functions.berhu_loss(gt_depth, depth, args.dataset)    
            elif args.loss=='L2':     
                loss_1 = loss_functions.l2_loss(gt_depth, depth, args.dataset)
            elif args.loss=='scale_inv':
                loss_1 = loss_functions.Scale_invariant_loss(gt_depth, depth, args.dataset)
            elif args.loss=='Multi_scale_inv':
                loss_1 = loss_functions.Multiscale_scale_inv_loss(gt_depth, depth)
            elif args.loss=='DORN':
                loss_1 = loss_functions.DORN_loss(gt_depth, pred_ord, target_c, args.dataset)
            else:
                raise "undefined loss"
        else:
            #original loss_1(unsupervised)
            loss_1 = loss_functions.photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                                    intrinsics, intrinsics_inv,
                                                                    depth, explainability_mask, pose,
                                                                    args.rotation_mode, args.padding_mode)

        if w2 > 0:
            loss_2 = loss_functions.explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        
        if args.loss == 'DORN':
            loss_3 = loss_functions.smooth_DORN_loss(pred_ord)
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        else:    
            loss_3 = loss_functions.smooth_loss(depth)
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            if w2 > 0:
                train_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
            if loss_3 > 0:
                train_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        if args.loss != 'DORN' and args.network != 'DORN':
            if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

                train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)

                for k, scaled_depth in enumerate(depth):
                    train_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                           tensor2array(disparities[k][0], max_value=None, colormap='bone'),
                                           n_iter)
                    train_writer.add_image('train Depth Output Normalized {}'.format(k),
                                           tensor2array(1/disparities[k][0], max_value=None),
                                           n_iter)
                    b, _, h, w = scaled_depth.size()
                    downscale = tgt_img.size(2)/h

                    tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
                    

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item() if loss_3 > 0 else 0])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = len(output_writers) > 0
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    poses = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1),6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        disp = disp_net(tgt_img)
        depth = 1/disp
        explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)

        loss_1 = photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                 intrinsics, intrinsics_inv,
                                                 depth, explainability_mask, pose,
                                                 args.rotation_mode, args.padding_mode)
        loss_1 = loss_1.item()
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask).item()
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth).item()

        if log_outputs and i < len(output_writers):  # log first output of every 100 batch
            if epoch == 0:
                for j,ref in enumerate(ref_imgs):
                    output_writers[i].add_image('val Input {}'.format(j), tensor2array(tgt_img[0]), 0)
                    output_writers[i].add_image('val Input {}'.format(j), tensor2array(ref[0]), 1)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(disp[0], max_value=None, colormap='bone'),
                                        epoch)
            output_writers[i].add_image('val Depth Output Normalized',
                                        tensor2array(1./disp[0], max_value=None),
                                        epoch)
            # log warped images along with explainability mask
            for j,ref in enumerate(ref_imgs):
                ref_warped = inverse_warp(ref[:1], depth[:1,0], pose[:1,j],
                                          intrinsics[:1], intrinsics_inv[:1],
                                          rotation_mode=args.rotation_mode,
                                          padding_mode=args.padding_mode)[0]

                output_writers[i].add_image('val Warped Outputs {}'.format(j),
                                            tensor2array(ref_warped),
                                            epoch)
                output_writers[i].add_image('val Diff Outputs {}'.format(j),
                                            tensor2array(0.5*(tgt_img[0] - ref_warped).abs()),
                                            epoch)
                if explainability_mask is not None:
                    output_writers[i].add_image('val Exp mask Outputs {}'.format(j),
                                                tensor2array(explainability_mask[0,j], max_value=1, colormap='bone'),
                                                epoch)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses[i * step:(i+1) * step] = pose.cpu().view(-1,6).numpy()
            step = args.batch_size * 3
            disp_unraveled = disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
    if log_outputs:
        prefix = 'valid poses'
        coeffs_names = ['tx', 'ty', 'tz']
        if args.rotation_mode == 'euler':
            coeffs_names.extend(['rx', 'ry', 'rz'])
        elif args.rotation_mode == 'quat':
            coeffs_names.extend(['qx', 'qy', 'qz'])
        for i in range(poses.shape[1]):
            output_writers[0].add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses[:,i], epoch)
        output_writers[0].add_histogram('disp_values', disp_values, epoch)
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Exp loss']


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)#;pdb.set_trace()
        
        #nyudepth have different structure of depth data
        if args.dataset=='nyu':
            depth = torch.squeeze(depth[:,0,:,:])# another data is just mask and this mask is calculated by depth==10 or depth==0

        # compute output
        if args.loss == 'DORN':
            pred, _ = disp_net(tgt_img)
            output_depth = torch.squeeze(utils.get_depth_sid(pred, ordinal_c=args.ordinal_c, dataset = args.dataset ))
        else:
            output_disp = disp_net(tgt_img)#;pdb.set_trace()
            if args.monodepth2:
                #pred_disp, output_depth = disp_to_depth(output_disp[:,0], min_depth=0.1, max_depth=100)
                output_depth = 1/output_disp[:,0]
                output_depth *= 5.4#scale factor of stereo training
            else:
                output_depth = 1/output_disp[:,0]
            
            if log_outputs and i < len(output_writers):
                if epoch == 0:
                    output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                    depth_to_show = depth[0]# not ok for nyu
                    output_writers[i].add_image('val target Depth',
                                                tensor2array(depth_to_show, max_value=10),
                                                epoch)
                    depth_to_show[depth_to_show == 0] = 1000
                    disp_to_show = (1/depth_to_show).clamp(0,10)
                    output_writers[i].add_image('val target Disparity Normalized',
                                                tensor2array(disp_to_show, max_value=None, colormap='bone'),
                                                epoch)

                output_writers[i].add_image('val Dispnet Output Normalized',
                                            tensor2array(output_disp[0], max_value=None, colormap='bone'),
                                            epoch)
                output_writers[i].add_image('val Depth Output',
                                            tensor2array(output_depth[0], max_value=3),
                                            epoch)

        if args.dataset=='nyu':
            #for the use of upsample result(nyu depth test dataset resolution is different from the ground truth)
            output_depth = torch.unsqueeze(output_depth,1)
            upsample = nn.UpsamplingBilinear2d(size=depth.size()[1:])#since here the depth has been squeezed
            output_depth = torch.squeeze(upsample(output_depth))

	#debug for the errors
	#**************************************
        # scale_factor = torch.div(torch.median(depth), torch.median(output_depth))
        # #scale_factor = np.median(depth)/np.median(output_depth)
        # #sl_tensor=torch.tensor(scale_factor)
        # #print()
        # errors.update(compute_errors(depth, output_depth*scale_factor))
	#**************************************
	#original
        errors.update(loss_functions.compute_errors(depth, output_depth, dataset=args.dataset, unsupervised=args.unsupervised))#;pdb.set_trace()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
# debug
#    print(errors.avg)
#    print(error_names)
    return errors.avg, error_names

# model loader for monodepth2
def load_model(pretrained_model, weights_folder):
    """Load model(s) from disk
    """
    load_weights_folder = os.path.expanduser(weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))

    for n in ["encoder", "depth"]:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        model_dict = pretrained_model[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        pretrained_model[n].load_state_dict(model_dict)

    # # loading adam state
    # optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
    # if os.path.isfile(optimizer_load_path):
    #     print("Loading Adam weights")
    #     optimizer_dict = torch.load(optimizer_load_path)
    #     self.model_optimizer.load_state_dict(optimizer_dict)
    # else:
    #     print("Cannot find Adam weights so Adam is randomly initialized")

if __name__ == '__main__':
    main()
