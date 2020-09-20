from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *

import argparse
import collections
import logging
import json
import re
import time

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

# this rnn encoder is modified  with word level attention
class RNNEncoder(nn.Module):
    def __init__(self,vocab_size, hidden_size, use_glove= True, glove_wts= None, word_embedding_size = 300, bidirectional=False,
               input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        if use_glove == True:
            word_embedding_size ==  300
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        if use_glove:
            self.embedding.load_state_dict({'weight': torch.tensor(glove_wts)})

        self.input_dropout = nn.Dropout(input_dropout_p)
        # self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size), 
        #                          nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.hidden_size = hidden_size
        self.num_dirs = 2 if bidirectional else 1
    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        # embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)
        # forward rnn
        output, hidden = self.rnn(embedded)
        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # (batch, max_len, hidden)
            output = output[recover_ixs]
        
        return output


class grounding_model(nn.Module):
    def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1, \
     coordmap=True, leaky=False, dataset=None, light=False):
        super(grounding_model, self).__init__()
        self.coordmap = coordmap
        self.light = light
        self.lstm = True
        self.emb_size = emb_size
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        ## Text model
        self.textdim, self.embdim= 600, 300
        self.textmodel = RNNEncoder(vocab_size=len(corpus),
                                    use_glove=True,
                                    glove_wts= corpus.get_glove_embed(),
                                    word_embedding_size=self.embdim,
                                    hidden_size=self.textdim//2,
                                    bidirectional=True,
                                    input_dropout_p=0.2,
                                    dropout_p=.1,
                                    variable_lengths=True)

        ## Mapping module
        self.mapping_visu = nn.Sequential(OrderedDict([
            ('0', ConvBatchNormReLU(1024 + 8, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('1', ConvBatchNormReLU(512 + 8, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('2', ConvBatchNormReLU(256 + 8, emb_size, 1, 1, 0, 1, leaky=leaky))
        ]))
        
        # Attn Softmax
        self.attn_sfmax = nn.Softmax(-1)
        self.mapping_lang = nn.Linear(self.textdim, emb_size)
        
        # visumodel + textmodel
        embin_size = emb_size*3
        
        self.fcn_emb = nn.Sequential(OrderedDict([
            ('0', torch.nn.Sequential(
                ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ('1', torch.nn.Sequential(
                ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ('2', torch.nn.Sequential(
                ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
        ]))
        self.fcn_out = nn.Sequential(OrderedDict([
            ('0', torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
            ('1', torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
            ('2', torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size//2, 3*5, kernel_size=1),)),
        ]))

    def text_attn(self, image_feat, lang_feat):
        """
        inputs : 
            image_feat = (B, C, H, W)
            lang_feat =  (B, T, C)
        outputs :
            lang_feat_attn = (B, H, W, C)
            image_lang_sf2 = (B, H, W)
        """

        B, C, H, W = image_feat.size()
        image_feat = image_feat.view(B, C, -1)

        image_feat_ = image_feat.transpose(1, 2) # B, 169, 512
        lang_feat_ = lang_feat.transpose(1, 2) # B, 512, 40

        image_lang = torch.matmul(image_feat_, lang_feat_) # B, 169, 40
        image_lang_sf1 = self.attn_sfmax(image_lang) # B, 169, 40
        image_lang_sum = torch.sum(image_lang, dim = -1) # B, 169
        image_lang_sf2 = torch.sigmoid(image_lang_sum).view(B, H, W) # B, 13, 13

        lang_feat_attn = torch.matmul(image_lang_sf1, lang_feat).transpose(1, 2).view(B, C, H, W)

        return lang_feat_attn, image_lang_sf2, image_lang_sf1


    def forward(self, image, word_id):
        
        ## Language Module
        max_len = (word_id != 0).sum(1).max().item()
        word_id = word_id[:, :max_len]
        raw_flang = self.textmodel(word_id)
        bs, mlen, embdim = raw_flang.shape
        flang = self.mapping_lang(raw_flang.view(-1, embdim)).view(bs, mlen, self.emb_size)
        
        ## Visual Module
        ## [1024, 13, 13], [512, 26, 26], [256, 52, 52]
        batch_size = image.size(0)
        raw_fvisu = self.visumodel(image)
        fvisu = []
        for ii in range(len(raw_fvisu)):
            raw_fvisu_ii = raw_fvisu[ii]
            coord = generate_coord(batch_size, raw_fvisu_ii.size(2), raw_fvisu_ii.size(3))
            f_visu_ii = torch.cat([raw_fvisu_ii, coord], dim=1)
            f_visu_ii = self.mapping_visu._modules[str(ii)](f_visu_ii)
            fvisu.append(f_visu_ii)

        
        # Vision-Language Attention 
        flang_attn = []
        attn_map = []
        image_lang_attn = []
        for ii in range(len(fvisu)):
            flang_attn_, attn_map_, image_lang_attn_ = self.text_attn(fvisu[ii], flang)
            flang_attn.append(flang_attn_)
            attn_map.append(attn_map_)
            image_lang_attn.append(image_lang_attn_)

        flangvisu = []
        for ii in range(len(fvisu)):
            fvisu_ii_attn = attn_map[ii].unsqueeze(1)*fvisu[ii]
            fvisu_ii_attn= F.normalize(fvisu_ii_attn, p=2, dim=1)
            fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)
            flang_attn[ii] = F.normalize(flang_attn[ii], p=2, dim=1)
            # print(fvisu[ii].shape, flang_attn[ii].shape)
            flangvisu.append(torch.cat([fvisu[ii], 
                                    fvisu_ii_attn, 
                                    flang_attn[ii]], dim=1))
            
        ## fcn
        intmd_fea, outbox = [], []
        for ii in range(len(fvisu)):
            intmd_fea.append(self.fcn_emb._modules[str(ii)](flangvisu[ii]))
            outbox.append(self.fcn_out._modules[str(ii)](intmd_fea[ii]))
        return outbox, attn_map, image_lang_attn
