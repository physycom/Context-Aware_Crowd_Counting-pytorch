# -*- coding: utf-8 -*-
# This scripts uses CACC net on a single image, compares result to ground truth and shows them side by side

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
import argparse

from cannet import CANNet
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="shtech", type=str, help="Dataset to use. \"venice\" or \"shtech\"")
parser.add_argument("--root", default="D:/Alex/", type=str, help="Root dir of datasets (with trailing backslash)")
parser.add_argument("--dev", default="cpu", type=str, help="Use cpu or gpu? For gpu cuda is mandatory")
args = parser.parse_args()

dataset_prefdir = args.root
if args.dataset == "venice":
    img_root= dataset_prefdir + 'venice/test_data/images/'
    gt_dmap_root= dataset_prefdir + 'venice/test_data/ground_truth/'
    img_name = '4895_000060'
    model_param_path='./checkpoints/venice_epoch_991.pth'
    data_mean = [0.531, 0.508, 0.474]
    data_std = [0.193, 0.189, 0.176]
elif args.dataset == "shtech":
    img_root= dataset_prefdir + 'ShanghaiTech/part_A_final/test_data/images/'
    gt_dmap_root= dataset_prefdir + 'ShanghaiTech/part_A_final/test_data/ground_truth/'
    img_name = 'IMG_32'
    model_param_path='./checkpoints/cvpr2019_CAN_SHHA_353.pth'
    data_mean = [0.409, 0.368, 0.359]
    data_std = [0.286, 0.274, 0.276]
else:
    raise Exception("Unknown dataset. Use \"venice\" or \"shtech\".")
    
img_path = img_root + img_name + '.jpg'
gt_path = gt_dmap_root + img_name + '.npy'

if args.dev == "cpu":
    device=torch.device("cpu")
elif args.dev == "gpu":   
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True # use cudnn?
else:
    raise Exception("Unknown device. Use \"cpu\" or \"gpu\".")
model=CANNet().to(device)
model.load_state_dict(torch.load(model_param_path))

model.eval()
torch.no_grad()

gt_dmap = np.load(gt_path)

img_orig = Image.open(img_path)
img = torchvision.transforms.ToTensor()(img_orig)
# -------- to remove once training is done with new norm -----------------------------
data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]
# ------------------------------------------------------------------------------------
img = torchvision.transforms.functional.normalize(img, mean=data_mean, std=data_std)
img = img.unsqueeze(0).to(device)

# forward propagation
et_dmap=model(img).detach()
et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()

fig, ax = plt.subplots(1,3, figsize=(15,4))
et = et_dmap.sum()
gt = gt_dmap.sum()
er = np.abs(et-gt)/gt
fig.suptitle("Total people ET: {:.0f} GT: {:.0f} Err: {:.3f}".format(et,gt, er))
ax[0].imshow(img_orig)
ax[1].imshow(gt_dmap,cmap=CM.hot)
ax[2].imshow(et_dmap,cmap=CM.hot)
for i in range(3):
    ax[i].set_label([])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
