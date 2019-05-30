import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import PIL.Image as Image
from matplotlib import cm as CM

#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.column_stack((np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.

    # -------------------------------------------------------
    # change root and paths to point the directory you need
    # -------------------------------------------------------
    # ---------- now generate the ShanghaiA's ground truth
    root = 'D:/Alex/ShanghaiTech/'
    part_A_train = os.path.join(root,'part_A_final/train_data','images')
    part_A_test = os.path.join(root,'part_A_final/test_data','images')
    part_B_train = os.path.join(root,'part_B_final/train_data','images')
    part_B_test = os.path.join(root,'part_B_final/test_data','images')

    # -------- uncomment to process venice dataset instead of part_B
    # root = 'D:/Alex/venice/'
    # part_B_train = os.path.join(root,'train_data','images')
    # part_B_test = os.path.join(root,'test_data','images')

    path_sets = [part_A_train,part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["image_info"][0,0][0,0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter_density(k)
        # plt.imshow(k,cmap=CM.jet)
        # break
        np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)

    # ---------- now see a sample from ShanghaiA
    # plt.imshow(Image.open(img_paths[0]))
    # gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth'))
    # plt.imshow(gt_file,cmap=CM.jet)
    # print(np.sum(gt_file))# don't mind this slight variation

    # ---------- now generate the ShanghaiB's ground truth. USE THIS PART FOR VENICE DATASET
    path_sets = [part_B_train,part_B_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        # --------- mat["annotation"] for venice,  # mat["image_info"][0,0][0,0][0] for Shanghai
        gt = mat["image_info"][0,0][0,0][0]
        # gt = mat["annotation"]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter(k,15)
        np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)
