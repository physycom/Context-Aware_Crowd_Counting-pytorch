import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as CM

import numpy as np

from cannet import CANNet

from PIL import Image

torch.backends.cudnn.enabled=True

dataset_prefdir = 'D:/Alex/'
img_root= dataset_prefdir + 'venice/test_data/images/'
gt_dmap_root= dataset_prefdir + 'venice/test_data/ground_truth/'

model_param_path='./checkpoints/venice_epoch_991.pth'

device=torch.device("cpu")
model=CANNet().to(device)
model.load_state_dict(torch.load(model_param_path))

#dataset=CrowdDataset(img_root,gt_dmap_root,gt_downsample=8)
#dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

model.eval()
torch.no_grad()

img_name = '4895_000060'
img_path = img_root + img_name + '.jpg'
gt_path = gt_dmap_root + img_name + '.npy'

gt_dmap = np.load(gt_path)

img_orig = Image.open(img_path)
img = torchvision.transforms.ToTensor()(img_orig)
img = torchvision.transforms.functional.normalize(img,
                                                  mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

img=img.unsqueeze(0).to(device)
#gt_dmap=gt_dmap.to(device)

# forward propagation
et_dmap=model(img).detach()
et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
print(et_dmap.shape)


fig, ax = plt.subplots(1,3, figsize=(15,4))
et = np.round(et_dmap.sum())
gt = np.round(gt_dmap.sum())
er = np.abs(et-gt)/gt
fig.suptitle("Total people ET: {} GT: {} Err: {}".format(et,gt,er))
ax[0].imshow(img_orig)
ax[1].imshow(gt_dmap,cmap=CM.jet)
ax[2].imshow(et_dmap,cmap=CM.jet)
plt.tight_layout()
plt.show()
