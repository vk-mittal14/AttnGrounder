import os
import sys
import argparse
import shutil
import time
import random
import gc
import json
from distutils.version import LooseVersion
import scipy.misc
import logging

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

from dataset.referit_loader import *
from model.grounding_model import *
from utils.parsing_metrics import *
from utils.utils import *


testset_path = "test.json"
with open(testset_path, "r") as t :
    given = json.load(t)

d = []
for i in given.keys():
    d.append((given[i]["img"], given[i]['command']))
    
torch.save(d, "./ln_data/talk2car_test.pth")


anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
anchors = [float(x) for x in anchors.split(',')]
anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

test_dataset = Talk2CarDataset(data_root="./ln_data/",
                testmode=True,
                split='test',
                imsize = 416,
                transform=input_transform,
                max_query_len=40,
                lstm=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=0)


model = grounding_model(corpus=test_dataset.corpus, light=False, emb_size=512, coordmap=True)
model = model.cuda()

best_model  =torch.load("saved_models/version_n2.0_continued_model_best_continued.pth.tar")
model.load_state_dict(best_model['model_state_dict'])
from munch  import Munch
args = {"size": 416, 
       "anchor_imsize": 416}
args = Munch(args)

def test_epoch(val_loader, model, size_average, mode='test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    end = time.time()
    model_preds = {}
    for batch_idx, (imgs, word_id, ratio, dw, dh, im_id) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()

        image = Variable(imgs)
        word_id = Variable(word_id)


        with torch.no_grad():
            pred_anchor, _ = model(image, word_id)
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))

        ## test: convert center+offset to box prediction
        pred_conf_list = []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(1,-1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,4)

        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(1):
            if max_loc[ii] < 3*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(1,3,grid,grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()

            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio


        ## convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)

        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        ## also revert image for visualization
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))

        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])

        model_preds[im_id[0]] = list(pred_bbox.detach().cpu().numpy().astype(int))
    
    return model_preds

preds = test_epoch(test_loader, model, False)



pred = {}
for ii in preds.keys():
    pred[ii] = list(preds[ii][0])
preds_to_out = {}
for i in range(len(given)):
    img_name = given[str(i)]['img']
    pred_bbox = list(map(int, pred[img_name]))
    to_out = [pred_bbox[0], pred_bbox[1], pred_bbox[2] - pred_bbox[0], pred_bbox[3] - pred_bbox[1]]
    command_tkn = given[str(i)]['command_token']
    preds_to_out[command_tkn] = to_out

with open("predictions.json", "w") as pr:
    json.dump(preds_to_out, pr)
    print("Saved Predictions in predictions.json")
