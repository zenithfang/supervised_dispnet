from scipy.misc.pilutil import imresize,imread
from scipy.ndimage.interpolation import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import pdb
from collections import Counter
# for depth ground truth
#from imageio import imsave


parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--prediction-dir", required=True, type=str, help="prediction")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#@torch.no_grad()
def main():
    args = parser.parse_args()
    #consistant with test_disp
    seq_length = 0

    dataset_dir = Path(args.dataset_dir)
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])]

    framework = test_framework_KITTI(dataset_dir, test_files, seq_length, args.min_depth, args.max_depth)

    print('{} files to test'.format(len(test_files)))
    errors = np.zeros((2, 7, len(test_files)), np.float32)

    #load prediction
    prediction = np.load(args.prediction_dir)#;pdb.set_trace()
    
    #setup for record ground truth
    gt_depth_record=np.zeros((len(test_files),375,1242))
    mask_record=np.zeros((len(test_files),375,1242))
    pred_record=np.zeros((len(test_files),375,1242))


    for j, sample in enumerate(tqdm(framework)):
        #read disparity from image
        pred_depth = prediction[j]#;pdb.set_trace()

        gt_depth = sample['gt_depth']
        
        pred_depth_zoomed = zoom(pred_depth,
                                 (gt_depth.shape[0]/pred_depth.shape[0],
                                  gt_depth.shape[1]/pred_depth.shape[1])
                                 ).clip(args.min_depth, args.max_depth)
       
        #*********************************************
        #record zoomed_pred
        
        h = pred_depth_zoomed.shape[0]
        w = pred_depth_zoomed.shape[1]
        if h >375 and w >1242:
        	pred_record[j,:,:] = pred_depth_zoomed[0:375,0:1242]
        elif h>375 and w <1242:
        	pred_record[j,:,0:w] = pred_depth_zoomed[0:375,:]
        elif h<375 and w >1242:
        	pred_record[j,0:h,:] = pred_depth_zoomed[:,0:1242]
        else:
        	pred_record[j,0:h,0:w] = pred_depth_zoomed
        #*********************************************

        if sample['mask'] is not None:
            pred_depth_zoomed = pred_depth_zoomed[sample['mask']]
            gt_depth = gt_depth[sample['mask']]#;pdb.set_trace()

        #*************************************************************
        
        #record gt_depth
        
        h = sample['gt_depth'].shape[0]
        w = sample['gt_depth'].shape[1]
        if h >375 and w >1242:
        	gt_depth_record[j,:,:] = sample['gt_depth'][0:375,0:1242]
        elif h>375 and w <1242:
        	gt_depth_record[j,:,0:w] = sample['gt_depth'][0:375,:]
        elif h<375 and w >1242:
        	gt_depth_record[j,0:h,:] = sample['gt_depth'][:,0:1242]
        else:
        	gt_depth_record[j,0:h,0:w] = sample['gt_depth']
        

        # #record zoomed_pred
        
        # h = pred_depth_zoomed.shape[0]
        # w = pred_depth_zoomed.shape[1]
        # if h >375 and w >1242:
        # 	pred_record[j,:,:] = pred_depth_zoomed[0:375,0:1242]
        # elif h>375 and w <1242:
        # 	pred_record[j,:,0:w] = pred_depth_zoomed[0:375,:]
        # elif h<375 and w >1242:
        # 	pred_record[j,0:h,:] = pred_depth_zoomed[:,0:1242]
        # else:
        # 	pred_record[j,0:h,0:w] = pred_depth_zoomed

        #record mask
        h = sample['mask'].shape[0]
        w = sample['mask'].shape[1]

        if h >375 and w >1242:
        	mask_record[j,:,:] = sample['mask'][0:375,0:1242]
        elif h>375 and w <1242:
        	mask_record[j,:,0:w] = sample['mask'][0:375,:]
        elif h<375 and w >1242:
        	mask_record[j,0:h,:] = sample['mask'][:,0:1242]
        else:
        	mask_record[j,0:h,0:w] = sample['mask']
        #*************************************************************

        scale_factor = np.median(gt_depth)/np.median(pred_depth_zoomed)
        scale_factor = 1
        errors[1,:,j] = compute_errors(gt_depth, pred_depth_zoomed*scale_factor)#;pdb.set_trace()

    mean_errors = errors.mean(2)
    error_names = ['abs_rel','sq_rel','rms','log_rms','a1','a2','a3']

    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))
    
    # output_file = Path('gt_depth')
    # np.save(output_file, gt_depth_record)
    # output_file = Path('pred_zoomed')
    # np.save(output_file, pred_record)
    # output_file = Path('mask')
    # np.save(output_file, mask_record)


#interpolate ground truth map
def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# for function that are outside

class test_framework_KITTI(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=100, step=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.calib_dirs, self.gt_files, self.img_files, self.displacements, self.cams = read_scene_data(self.root, test_files, seq_length, step)

    def __getitem__(self, i):
        tgt = imread(self.img_files[i][0]).astype(np.float32)
        depth = generate_depth_map(self.calib_dirs[i], self.gt_files[i], tgt.shape[:2], self.cams[i])
        return {'tgt': tgt,
                'ref': [imread(img).astype(np.float32) for img in self.img_files[i][1]],
                'path':self.img_files[i][0],
                'gt_depth': depth,
                'displacement': np.array(self.displacements[i]),
                'mask': generate_mask(depth, self.min_depth, self.max_depth)
                }

    def __len__(self):
        return len(self.img_files)


###############################################################################
#  EIGEN

def getXYZ(lat, lon, alt):
    """Helper method to compute a R(3) pose vector from an OXTS packet.
    Unlike KITTI official devkit, we use sinusoidal projection (https://en.wikipedia.org/wiki/Sinusoidal_projection)
    instead of mercator as it is much simpler.
    Initially Mercator was used because it renders nicely for Odometry vizualisation, but we don't need that here.
    In order to avoid problems for potential other runs closer to the pole in the future,
    we stick to sinusoidal which keeps the distances cleaner than mercator (and that's the only thing we want here)
    See https://github.com/utiasSTARS/pykitti/issues/24
    """
    er = 6378137.  # earth radius (approx.) in meters
    scale = np.cos(lat * np.pi / 180.)
    tx = scale * lon * np.pi * er / 180.
    ty = er * lat * np.pi / 180.
    tz = alt
    t = np.array([tx, ty, tz])
    return t


def get_displacements(oxts_root, indices, tgt_index):
    """gets mean displacement magntidue between middle frame and other frames, this is, to a scaling factor
    the mean output PoseNet should have for translation. Since the scaling is the same factor for depth maps and
    for translations, it will be used to determine how much predicted depth should be multiplied to."""
    first_pose = None
    displacement = 0
    if len(indices) == 0:
        return 0
    reordered_indices = [indices[tgt_index]] + [*indices[:tgt_index]] + [*indices[tgt_index + 1:]]
    for index in reordered_indices:
        oxts_data = np.genfromtxt(oxts_root/'data'/'{:010d}.txt'.format(index))
        lat, lon, alt = oxts_data[:3]
        pose = getXYZ(lat, lon, alt)
        if first_pose is None:
            first_pose = pose
        else:
            displacement += np.linalg.norm(pose - first_pose)
    return displacement / max(len(indices - 1), 1)


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    data_root = Path(data_root)
    gt_files = []
    calib_dirs = []
    im_files = []
    cams = []
    displacements = []
    demi_length = (seq_length - 1) // 2
    shift_range = step * np.arange(-demi_length, demi_length + 1)

    print('getting test metadata ... ')
    for sample in tqdm(test_list):
        tgt_img_path = data_root/sample
        date, scene, cam_id, _, index = sample[:-4].split('/')

        scene_length = len(tgt_img_path.parent.files('*.png'))

        ref_indices = shift_range + np.clip(int(index), step*demi_length, scene_length - step*demi_length - 1)

        ref_imgs_path = [tgt_img_path.dirname()/'{:010d}.png'.format(i) for i in ref_indices]
        vel_path = data_root/date/scene/'velodyne_points'/'data'/'{}.bin'.format(index[:10])

        if tgt_img_path.isfile():
            gt_files.append(vel_path)
            calib_dirs.append(data_root/date)
            im_files.append([tgt_img_path,ref_imgs_path])
            cams.append(int(cam_id[-2:]))
            displacements.append(get_displacements(data_root/date/scene/'oxts', ref_indices, demi_length))
        else:
            print('{} missing'.format(tgt_img_path))
    # print(num_probs, 'files missing')

    return calib_dirs, gt_files, im_files, displacements, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:,3] = 1
    return points


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2,interp=False):#interp true just to produce ground truth
    # load calibration files
    cam2cam = read_calib_file(calib_dir/'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir/'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    
    #add for produce smooth ground truth
    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp(im_shape, velo_pts_im)
        return depth_interp
    else:
        return depth

#interpolate ground truth map
def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask


if __name__ == '__main__':
    main()

