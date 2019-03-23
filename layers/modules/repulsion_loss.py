#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jilong Wang
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@Description: file content
@Date: 2019-03-22 15:41:28
@LastEditTime: 2019-03-23 15:06:47
'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import IoG, decode_new
import sys


class RepulsionLoss(nn.Module):

    def __init__(self, use_gpu=True, sigma=0., variance=None):
        super(RepulsionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = variance
        self.sigma = sigma
        
    # TODO 
    def smoothln(self, x, sigma=0.):        
        pass

    def forward(self, loc_data, ground_data, prior_data):
        print(loc_data.size(),prior_data.size())
        decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
        iog = IoG(ground_data, decoded_boxes)
        # sigma = 1
        # loss = torch.sum(-torch.log(1-iog+1e-10))  
        # sigma = 0
        loss = torch.sum(iog)          
        return loss