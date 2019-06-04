import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as CM

import numpy as np

from cannet import CANNet
from my_dataset import CrowdDataset

from PIL import Image

torch.backends.cudnn.enabled=True

dataset_prefdir = 'E:/Alessandro/'
img_root= dataset_prefdir + 'ShanghaiTech/part_A_final/test_data/images'
gt_dmap_root= dataset_prefdir + 'ShanghaiTech/part_A_final/test_data/ground_truth'

model_param_path='./checkpoints/part_A_final_epoch_588.pth'

device=torch.device("cuda")
model=CANNet().to(device)
model.load_state_dict(torch.load(model_param_path))

#dataset=CrowdDataset(img_root,gt_dmap_root,gt_downsample=8)
#dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

model.eval()

img_name = 'IMG_100'
img_path = 'E:\\Alessandro\\ShanghaiTech\\part_A_final\\test_data\\images\\' + img_name + '.jpg'
gt_path = 'E:\\Alessandro\\ShanghaiTech\\part_A_final\\test_data\\ground_truth\\' + img_name + '.npy'

gt_dmap = np.load(gt_path)

img_orig = Image.open(img_path)
img = torchvision.transforms.ToTensor()(img_orig).unsqueeze(0)

img=img.to(device)
#gt_dmap=gt_dmap.to(device)

# forward propagation
et_dmap=model(img).detach()
et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
print(et_dmap.shape)


fig, ax = plt.subplots(1,2)
et = np.round(et_dmap.sum())
gt = np.round(gt_dmap.sum())
er = np.abs(et-gt)/gt
fig.suptitle("Total stronzi ET: {} GT: {} Err: {}".format(et,gt,er))
ax[0].imshow(img_orig)
ax[1].imshow(et_dmap,cmap=CM.jet)
plt.show()
