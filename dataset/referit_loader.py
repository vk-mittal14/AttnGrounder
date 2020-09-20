# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import pickle
import random
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import utils
from utils import Corpus

import argparse
import collections
import logging
import json
import re
from PIL import Image, ImageDraw


from utils.transforms import letterbox, random_affine

sys.modules['utils'] = utils

cv2.setNumThreads(0)


def bbox_randscale(bbox, miniou=0.75):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    scale_shrink = (1-math.sqrt(miniou))/2.
    scale_expand = (math.sqrt(1./miniou)-1)/2.
    w1,h1 = random.uniform(-scale_expand, scale_shrink)*w, random.uniform(-scale_expand, scale_shrink)*h
    w2,h2 = random.uniform(-scale_shrink, scale_expand)*w, random.uniform(-scale_shrink, scale_expand)*h
    bbox[0],bbox[2] = bbox[0]+w1,bbox[2]+w2
    bbox[1],bbox[3] = bbox[1]+h1,bbox[3]+h2
    return bbox


class Talk2CarDataset(data.Dataset):
    def __init__(self, data_root, imsize=416,
                 transform=None, augment=False, return_idx=False, testmode=False,
                 split='train', max_query_len=40, lstm=True):
        self.images = []
        self.data_root = data_root
        self.imsize = imsize
        self.query_len = max_query_len
        self.lstm = lstm

        glove_path = osp.join(self.data_root, "glove_dict_6B.300.pkl")
        glove = pickle.load(open(glove_path, "rb"))
        self.corpus = Corpus(glove)

        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.augment=augment
        self.im_dir = osp.join(self.data_root, 'images')

        if osp.isfile(osp.join(self.data_root, "corpus.pth")):
            print("Loading Saved Corpus")
            corpus_path = osp.join(self.data_root, 'corpus.pth')
            self.corpus = torch.load(corpus_path)
        else:
            print("=> Building Corpus")
            self.corpus.load_file(osp.join(self.data_root, "talk2car.txt"))
            print("=> Saving Corpus")
            torch.save(self.corpus, osp.join(self.data_root, "corpus.pth"))

        imgset_file = f'talk2car_{split}.pth'
        imgset_path = osp.join(self.data_root, imgset_file)
        self.images = torch.load(imgset_path)


    def pull_item(self, idx):
        if self.testmode:
            img_file, phrase = self.images[idx]
        else:
            img_file, bbox, phrase = self.images[idx]
            ## box format: to x1y1x2y2
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]

        img_path = osp.join(self.im_dir, img_file)
        img = np.array(Image.open(img_path))
        if self.testmode:
            return img, phrase
        else:
            return img, phrase, bbox

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.testmode:
            img, phrase= self.pull_item(idx)
        else:
            img, phrase, bbox = self.pull_item(idx)

        phrase = phrase.lower()
        if self.augment:
            augment_flip, augment_hsv, augment_affine = True,True,True

        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        if self.augment:
            ## random horizontal flip
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            ## random affine transformation
            if augment_affine:
                img, _, bbox, M = random_affine(img, None, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
        else:   ## should be inference, or specified training
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            if self.testmode == False: 
                bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
                bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh

        ## Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        
        phrase = self.tokenize_phrase(phrase)
        if self.testmode == False: 
            object_map = Image.new("L", (img.size(1), img.size(2)))
            object_map_ = ImageDraw.Draw(object_map)
            # print(bbox.shape)
            # print(img.shape)
            bbox_ = list(map(int, list(bbox)))
            object_map_.rectangle(bbox_, fill ="white")
            object_map_8 = (np.array(object_map.resize((img.size(1)//8, img.size(2)//8), Image.BILINEAR))> 1).astype(int)
            object_map_16 = (np.array(object_map.resize((img.size(1)//16, img.size(2)//16), Image.BILINEAR))> 1).astype(int)
            object_map_32 = (np.array(object_map.resize((img.size(1)//32, img.size(2)//32), Image.BILINEAR))> 1).astype(int)
        if self.testmode:
            return img, np.array(phrase, dtype=int), \
                 np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            return img, object_map_32, object_map_16, object_map_8, np.array(phrase, dtype=int), \
            np.array(bbox, dtype=np.float32)