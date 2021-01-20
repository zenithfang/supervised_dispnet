import matplotlib.pyplot as plt
import numpy as np
import pdb

reso = False

y_D_pixel_416 = np.load('abs_test_pixel.npy');pdb.set_trace()
# check certain supervision result
# y_test = y_D_416[0,:]#for direct supervision
# y_test = y_S_416[:,0]
y_test = y_D_pixel_416[:,0]

# max_5_error_index = np.argpartition(y_test, -5)[-5:]
# max_5_error = y_test[max_5_error_index]
# min_5_error_index = np.argpartition(y_test, 5)[:5]
# min_5_error = y_test[min_5_error_index]#;pdb.set_trace()



#max error and corresponding index
#array([382, 395, 388, 385, 174])
#array([0.23704123, 0.25923422, 0.27255267, 0.29910684, 0.3097115 ],dtype=float32)#D result
#array([0.24553713, 0.26688939, 0.25118491, 0.16997431, 0.25148839])#S result
#array([0.11141012, 0.34819573, 0.14325334, 0.19597265, 0.23082088])#M result
#min error and corresponding index
#array([405, 660, 216, 654, 293])#array([0.04548579, 0.03550705, 0.04592678, 0.04738897, 0.04654082],dtype=float32)

#result of y_S_416#similar max error images

#max error and corresponding index
#array([382, 388, 395, 383, 174])array([0.24553713, 0.25118491, 0.26688939, 0.3148692 , 0.25148839])
#min error and corresponding index
#array([216,   2,  52, 293,   4])array([0.04179476, 0.04006128, 0.04325297, 0.0440316 , 0.04586333])

#result of y_M_416

#max error and corresponding index
#array([174, 685, 164, 374, 395])array([0.23082088, 0.24455129, 0.25492683, 0.2615391 , 0.34819573])
#min error and corresponding index
#array([405,  52,   0,   3,  11])array([0.04185434, 0.04804014, 0.04179324, 0.0492242 , 0.05022766])

#x = np.linspace(0, y_DMS_1024.shape[1], y_DMS_1024.shape[1], endpoint=True)#;pdb.set_trace()

#plt.hold(True)
if reso:
	# plt.plot(x, y_DMS_1024[0,:], 'b', label= 'dms_1024')
	# plt.plot(x, y_DMS_416[0,:], 'r', label= 'dms_416')
	plt.scatter(y_DMS_1024[0,:], y_DMS_416[0,:], c='r', label= 'dms_416')
	plt.title("Abs Rel resolution comparison")

	plt.xlabel("dms_1024")
	plt.ylabel("dms_416")
	plt.legend(loc='upper left')
	plt.savefig("Direct_reso_com_graph.pdf")
else:
    # plt.plot(x, y_DMS_416[0,:], 'r', label= 'dms_416')
    # plt.plot(x, y_D_416[0,:], 'b', label= 'd_416')
    # plt.plot(x, y_S_416[:,0], 'g', label= 's_416')
    # plt.plot(x, y_M_416[:,0], 'm', label= 'm_416')
    fig, axs = plt.subplots(2, 2, sharex='all')

    # add a big axes, hide frame(this is for sharex and sharey)
    fig.add_subplot(111, frameon = False)
	# hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("direct supervision")#set up common label

    axs[0, 0].scatter(y_D_416[0,:], y_S_416[:,0], c='m', s=5)
    axs[0, 0].set_title('direct vs stereo')
    axs[0, 0].set_ylabel('stereo')
    axs[0, 0].set_xlabel('a')
    axs[0, 1].scatter(y_D_416[0,:], y_M_416[:,0], c='m', s=5)
    axs[0, 1].set_title('direct vs video')
    axs[0, 1].set_ylabel('video')
    axs[0, 1].set_xlabel('b')
    axs[1, 0].scatter(y_D_416[0,:], y_D_res50_416[0,:], c='m', s=5)
    axs[1, 0].set_title('direct with different models')
    axs[1, 0].set_ylabel('different models')
    axs[1, 0].set_xlabel('c')
    axs[1, 1].scatter(y_D_416[0,:], y_D_seed_416[0,:], c='m', s=5)
    axs[1, 1].set_title('direct with different seeds')
    axs[1, 1].set_ylabel('diferent seeds')
    axs[1, 1].set_xlabel('d')
    # plt.title("Abs Rel Supervised method comparison")
    plt.tight_layout()
    #plt.title("Abs Rel method comparison")
    plt.savefig("Sup_com_graph.pdf")

plt.show()


