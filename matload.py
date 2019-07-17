# -*- coding: utf-8 -*
# This script loads estimated dc and gt locations and overlaps them to the tested images.
# gt = groundtruth
# dc = density colormap

import scipy.io
import numpy as np
import cv2
import os

path = "D:/Alex/venice/test_data/"
path_mat = path + "ground_truth/"
path_gt = path + "ground_truth/"
path_img = path + "images/"
norm = 0.005 # normalization for the dc
density_weight = 0.7 # opacity of dc overlap
draw_cross = False # choose whether to draw gt crosses or not

for _,_,c in os.walk(path_img):
    for file in c:
        imgn = file.split('.')[0]
        mat = np.load(path_mat + imgn + ".npy")
        image = cv2.cvtColor(cv2.imread(path_img + imgn + ".jpg", 0), cv2.COLOR_GRAY2BGR)
        
        # normalization gets recalibrated if density max is larger than current norm
        if mat.max() > norm:
            norm = mat.max()
        dmap = cv2.applyColorMap(np.array(mat/norm * 255, dtype = np.uint8), 11) # last parameter is colormap. 11 for hot, 2 for jet
        res = cv2.addWeighted(image, 1 - density_weight, dmap, density_weight, 0)
        
        if draw_cross:
            gt = scipy.io.loadmat(path_gt + imgn + ".mat")
            points = gt["annotation"]  # ["image_info"][0,0][0,0][0] for Shanghai
            for p in points:
                res = cv2.drawMarker(res, tuple(p), color = (255,255,255), markerSize = 5)
            
        print(imgn + " ) tot gt: " + str(len(points)) + ", tot detect:" + str(mat.sum().round()))
        cv2.imwrite("./results/"+imgn+".jpg",res)