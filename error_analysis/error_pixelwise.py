import matplotlib.pyplot as plt
import numpy as np
import pdb
from path import Path

load = 'M'
sample =False
#M monocular
#S stereo
#D direct
np.random.seed(10)
y_M_pixel_416 = np.load('/scratch_net/airfox_second/zfang/monodepth2-master/abs_test_pixel_mono.npy')#;pdb.set_trace()
y_S_pixel_416 = np.load('/scratch_net/airfox_second/zfang/monodepth2-master/abs_test_pixel_stereo.npy')
y_D_pixel_416 = np.load('abs_test_pixel.npy')

# for i in range(len(y_M_pixel_416)):
for i in range(len(y_S_pixel_416)):
    valid = y_M_pixel_416[i]>0

    gt_height, gt_width = valid.shape[:2]
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(valid.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    valid = np.logical_and(valid, crop_mask)

    if i == 0:
        # y_M_pixel store all of the processing data
        # y_M_pixel = y_M_pixel_416[i][valid]# one pic with about 17k valid points
        # y_S_pixel = y_S_pixel_416[i][valid]
        # y_D_pixel = y_D_pixel_416[i][valid]
        
        y_M_pixel = y_M_pixel_416[i][valid]# one pic with about 17k valid points
        y_S_pixel = y_S_pixel_416[i][valid]
        y_D_pixel = y_D_pixel_416[i][valid]

        # y_M_pixel_index = valid# one pic with about 17k valid points
        # y_S_pixel_index = valid
        # y_D_pixel_index = valid

        if sample:
        # choose only 100 points from each pics
            y_M_pixel = y_M_pixel[np.random.choice(y_M_pixel.shape[0], 100, replace=False)]
            y_S_pixel = y_S_pixel[np.random.choice(y_S_pixel.shape[0], 100, replace=False)]
            y_D_pixel = y_D_pixel[np.random.choice(y_D_pixel.shape[0], 100, replace=False)]

    else:
        # fetch current data
        y_M_pixel_current = y_M_pixel_416[i][valid]# one pic with about 17k valid points
        y_S_pixel_current = y_S_pixel_416[i][valid]
        y_D_pixel_current = y_D_pixel_416[i][valid]
        
        # sample data out
        if sample:
            y_M_pixel_current = y_M_pixel_current[np.random.choice(y_M_pixel_current.shape[0], 100, replace=False)]
            y_S_pixel_current = y_S_pixel_current[np.random.choice(y_S_pixel_current.shape[0], 100, replace=False)]
            y_D_pixel_current = y_D_pixel_current[np.random.choice(y_D_pixel_current.shape[0], 100, replace=False)]

        # append data
        y_M_pixel = np.append(y_M_pixel, y_M_pixel_current)
        y_S_pixel = np.append(y_S_pixel, y_S_pixel_current)
        y_D_pixel = np.append(y_D_pixel, y_D_pixel_current);pdb.set_trace()

# fig, axs = plt.subplots(3, 1, sharex='all')

# axs[0].scatter(y_D_pixel, y_S_pixel, c='m', s=0.25)
# axs[0].set_title('direct vs stereo')
# axs[0].set_ylabel('stereo')
# axs[0].set_xlabel('a')
# axs[1].scatter(y_D_pixel, y_M_pixel, c='m', s=0.25)
# axs[1].set_title('direct vs video')
# axs[1].set_ylabel('video')
# axs[1].set_xlabel('b')
# axs[2].scatter(y_S_pixel, y_M_pixel, c='m', s=0.25)
# axs[2].set_title('stereo vs video')
# axs[2].set_ylabel('video')
# axs[2].set_xlabel('c')
# plt.tight_layout()

# plot histogram
plt.figure(1);pdb.set_trace()
# wait for more possible modification like the threshold of x axis and also visualization of single image(mainly for rank of data in single image)

for i in range(1000):
    pass

output_file = Path('abs_test_pixel_stereo')
np.save(output_file, y_S_pixel)

# filter out the abs that larger than
y_M_pixel_filter = y_M_pixel[y_M_pixel<0.3]

n, bins, patches = plt.hist(y_M_pixel, 50, density=True, facecolor='g', alpha=0.75)
plt.xlabel('abs rel')
plt.ylabel('Probability')
plt.title('histogram of video model result')
plt.axis([0, 0.3, 0, 1])
plt.grid(True)
plt.savefig("M_pixel_hist.pdf")


# plt.figure(2)
# n, bins, patches = plt.hist(y_S_pixel, 50, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('abs rel')
# plt.ylabel('Probability')
# plt.title('Histogram of stereo model result')
# plt.savefig("S_pixel_hist.pdf")

# plt.figure(3)
# n, bins, patches = plt.hist(y_D_pixel, 50, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('abs rel')
# plt.ylabel('Probability')
# plt.title('Histogram of direct model result')
# plt.savefig("D_pixel_hist.pdf")

# y_M_pixel_416[0]

# # single image analysis
# # plt.scatter(y_D_pixel, y_S_pixel, c='m', s=0.25)
# plt.scatter(y_D_pixel, y_M_pixel, c='m', s=0.25)
# plt.xlim(0, 0.3)
# plt.ylim(0, 0.3)
# plt.savefig("Sup_com_pixel_graph_D_M.pdf")
# #plt.show()


