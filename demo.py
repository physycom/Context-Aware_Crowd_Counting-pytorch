# -*- coding: utf-8 -*-
# This scripts uses CACC net on a video file / video cam to estimate density map
# and crowd count and shows them overlapped to the video.

import torch
import torchvision
import numpy as np
import cv2
import argparse

from cannet import CANNet

#debug purpose
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="C:/Users/alex/Desktop/scripts/cacc-physycom/checkpoints/venice_epoch_991.pth", type=str, help="Weights file to use")
    parser.add_argument("--video", default="D:/Alex/venice/test_data/videos/4895.mp4", type=str, help="Video file to elapse")
    parser.add_argument("--device", default="cpu", type=str, help="Use cpu or gpu? For gpu cuda is mandatory")
    parser.add_argument("--opacity", default=0.7, type=float, help="Opacity value for the density map overlap")
    args = parser.parse_args()
    
    # setup the model
    if args.device == "cpu":
        device=torch.device("cpu")
    elif args.device == "gpu":   
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = True # use cudnn?
    else:
        raise Exception("Unknown device. Use \"cpu\" or \"gpu\".")
    model = CANNet().to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    torch.no_grad()
    
    # open the video stream / file
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        t1 = time.clock()
        _, frame = cap.read()
        
        # convert to pytorch tensor and normalize
        tensor = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # ---------- to change after new norm training ------------------------
        tensor = torchvision.transforms.functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ---------------------------------------------------------------------
        tensor = tensor.unsqueeze(0).to(device)
        t2 = time.clock()
        # forward propagation
        ed_map = model(tensor).detach()
        t3 = time.clock()
        ed_map = ed_map.squeeze(0).squeeze(0).cpu().numpy()
        ed_map*=(ed_map>0) # sets to 0 the negative values
        # converts frame to grayscale and density map to color map
        gray = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
        cmap = cv2.applyColorMap(np.array(ed_map/ed_map.max() * 255, dtype = np.uint8), 11) # last parameter is colormap. 11 for hot, 2 for jet
        # resize density map to frame size and overlap them
        cmap = cv2.resize(cmap, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_NEAREST)
        res = cv2.addWeighted(gray, 1 - args.opacity, cmap, args.opacity, 0)
        # write total people count on image and show results
        cv2.putText(res, str(int(ed_map.sum()+0.5)), (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Density map", res)
        t4 = time.clock()
        print("forward prop:" + str(t3-t2))
        print("whole frame: " + str(t4-t1))
        # wait for escape key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
