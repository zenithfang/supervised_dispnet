import torch
import torchvision.transforms
from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import datetime
#from models import DispNetS
import models
from utils import tensor2array, get_depth_sid
import pdb
from PIL import Image, ImageEnhance

import networks
import os

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--network", default='disp_vgg', type=str, help="network type")
parser.add_argument('--imagenet-normalization', action='store_true', help='use imagenet parameter for normalization.')
parser.add_argument("--monodepth2", action='store_true', help="to inference monodepth2 model")

parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    # load ground truth avg for scale
    # scale_factor = np.load('gt_avg_test.npy')
    
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
        load_model(pretrained_model = mono2_models, weights_folder = args.pretrained)
        #construct this disp_net to be compatiable with existing framework
        disp_net = models.monodepth2(encoder = mono2_models["encoder"], decoder = mono2_models["depth"])
    else:
        if args.network=='dispnet':
            disp_net = models.DispNetS().to(device)
        elif args.network=='disp_res':
            disp_net = models.Disp_res().to(device)
        elif args.network=='disp_vgg':
            disp_net = models.Disp_vgg_feature().to(device)
        elif args.network=='disp_vgg_BN':
            disp_net = models.Disp_vgg_BN().to(device)
        elif args.network=='FCRN':
            disp_net = models.FCRN().to(device)     
        elif args.network=='ASPP':
            disp_net = models.deeplab_depth().to(device)   
        elif args.network=='disp_vgg_BN_DORN':
            disp_net = models.Disp_vgg_BN_DORN().to(device)   
        else:
            raise "undefined network"
    
    if not args.monodepth2:# monodepth2 has already read weight
        weights = torch.load(args.pretrained)
        disp_net.load_state_dict(weights['state_dict'])

    disp_net.eval()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    net_name = Path(args.network)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)/net_name/timestamp
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))
    
    #save max for get depth from picture
    pred_max=np.zeros(len(test_files))

    for j, file in enumerate(tqdm(test_files)):

        img = imread(file).astype(np.float32)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        #for different normalize method
        if args.imagenet_normalization:
            if args.monodepth2:
            # this date normalize of monodepth2 is written inside the dispnet
            # thus we just use this 0 and 1 that do not change data
                normalize = torchvision.transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1])
            else:
                normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
 
        tensor_img = torch.from_numpy(img)#.unsqueeze(0)
        # tensor_img = ((tensor_img/255 - 0.5)/0.2).to(device)% why it is 0.2
        tensor_img = normalize(tensor_img/255).unsqueeze(0).to(device)# consider multiply by 2.5 to compensate

        if args.network=='DORN' or args.network == 'disp_vgg_BN_DORN':
            pred_d, pred_ord = disp_net(tensor_img)
            #pred_depth = torch.squeeze(get_depth_sid(pred_d))#.cpu().numpy()#;pdb.set_trace()
            pred_depth = get_depth_sid(pred_d)[0]
            output = 1/pred_depth#;pdb.set_trace()
        else:
            output = disp_net(tensor_img)[0]
        
        #add normalize from median of ground truth 
        # pred = disp_net(tensor_img).cpu().numpy()[0,0];#pdb.set_trace()
        # output = output*(scale_factor[j]/np.median(pred))
        # #save pred_max for recover depth from pic
        # pred_max[j] = np.amax(pred) 

        if args.output_disp:
            crop = [0.40810811 * args.img_height, 0.99189189 * args.img_height,
                    0.03594771 * args.img_width,  0.96405229 * args.img_width]
            crop = [int(i) for i in crop]
            resize_output = output[:,crop[0]:crop[1],crop[2]:crop[3]]
            disp = (255*tensor2array(resize_output, max_value=None, colormap='bone', channel_first=False)).astype(np.uint8)
            #max_value 50 or 80 is like the clamp(this colormap is significantly influenced by small value, thus sometimes 
            #the relative value that divide by max depth would be influenced by the max depth predicted over the 
            #middle of lane(due to the imprecise max prediction))
            
            #original one
            #disp = (255*tensor2array(output, max_value=None, colormap='bone', channel_first=False)).astype(np.uint8)

            #check comparison
            #disp = (tensor2array(output, max_value=None, colormap='bone', channel_first=False)).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(j,file.ext), disp)

            #add contrast
            im=Image.open(output_dir/'{}_disp{}'.format(j,file.ext))
            enhancer = ImageEnhance.Contrast(im)
            enhanced_im=enhancer.enhance(4.0)
            enhanced_im.save(output_dir/'{}_en{}'.format(j,file.ext))

        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow', channel_first=False)).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file.namebase,file.ext), depth)
    
    # output_file = Path('pred_max_test')
    # np.save(output_file, pred_max)

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

def generate_mask(height, width):
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    crop = np.array([0.40810811 * height, 0.99189189 * height,
                     0.03594771 * width,  0.96405229 * width]).astype(np.int32)
    crop_mask = np.zeros((height, width))
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    crop_mask = torch.from_numpy(crop_mask).unsqueeze(0).byte().to(device)
    return crop_mask

if __name__ == '__main__':
    main()
