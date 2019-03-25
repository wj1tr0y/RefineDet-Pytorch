#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jilong Wang
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@Description: file content
@Date: 2019-03-22 15:41:28
@LastEditTime: 2019-03-25 18:27:58
'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import IoG, decode, jaccard
import sys
import math


class RepulsionLoss(nn.Module):

    def __init__(self, use_gpu=True, sigma=0., variance=None):
        super(RepulsionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = variance
        self.sigma = sigma
        
    def repgt(self, pred_boxes, gt_bboxes, pos_idx):

        sigma_repgt = 0.9
        loss_repgt=torch.zeros(pred_boxes.shape[0]).cuda()                                                                                                                                                      
        for i in range(pred_boxes.shape[0]):                                                                                                                                                                                       
            boxes = Variable(pred_boxes[i][pos_idx[i]].view(-1,4)).cuda()
            gt = Variable(gt_bboxes[i].view(-1,4)).cuda()
            num_repgt = 0
            repgt_smoothln=0
            
            if boxes.shape[0]>0:
                overlaps = jaccard(boxes, gt)
                max_overlaps, argmax_overlaps = torch.max(overlaps,1)
                for j in range(overlaps.shape[0]):
                    overlaps[j,argmax_overlaps[j]] = 0
                
                max_overlaps, argmax_overlaps = torch.max(overlaps,1)
                for j in range(max_overlaps.shape[0]):
                    if max_overlaps[j]>0:
                        num_repgt += 1
                        iog = IoG(boxes[j], gt[argmax_overlaps[j]])
                        if iog>sigma_repgt:
                            repgt_smoothln += ((iog-sigma_repgt) / (1-sigma_repgt) - math.log(1 - sigma_repgt))
                        elif iog<=sigma_repgt:
                            repgt_smoothln += -math.log(1-iog)
            if num_repgt>0:
                loss_repgt[i] = repgt_smoothln / num_repgt
                
        return loss_repgt			

    def repbox(self, pred_boxes, gt_bboxes, pos_idx):

        sigma_repbox = 0
        loss_repbox=torch.zeros(pred_boxes.shape[0]).cuda()

        for i in range(pred_boxes.shape[0]):
            
            boxes = Variable(pred_boxes[i][pos_idx[i]].view(-1, 4)).cuda()
            gt = Variable(gt_bboxes[i].view(-1, 4)).cuda()
    
            num_repbox = 0
            repbox_smoothln = 0
            if boxes.shape[0]>0:
                overlaps = jaccard(boxes, boxes)
                # for j in range(overlaps.shape[0]):
                #     for z in range(overlaps.shape[1]):
                #         if z>=j:
                #             overlaps[j,z]=0
                #         elif int(torch.sum(gt[j]==gt[z]))==4:
                #             overlaps[j,z]=0

                iou=overlaps[overlaps>0]
                for j in range(iou.shape[0]):
                    num_repbox+=1
                    if iou[j]<=sigma_repbox:
                        repbox_smoothln += -math.log(1-iou[j])
                    elif iou[j]>sigma_repbox:
                        repbox_smoothln += ((iou[j]-sigma_repbox) / (1-sigma_repbox) - math.log(1-sigma_repbox))

            if num_repbox>0:
                loss_repbox[i] = repbox_smoothln / num_repbox
                
        return loss_repbox
                    
    def repulsion(self, loc_data, gt_bboxes, priors, pos_idx):                                             
                                                            
        pred_boxes = torch.FloatTensor(loc_data.size()).cuda()
        for j in range(loc_data.size()[0]):
            pred_boxes[j] = decode(loc_data[j], priors, self.variance)

        loss_repgt = self.repgt(pred_boxes, gt_bboxes, pos_idx)
        loss_repbox = self.repbox(pred_boxes, gt_bboxes, pos_idx)

        return loss_repgt, loss_repbox

    def forward(self, loc_data, ground_data, prior_data, pos_idx):
        
        loss_repgt, loss_repbox = self.repulsion(loc_data, ground_data, prior_data, pos_idx)
        loss = torch.sum(loss_repgt) + torch.sum(loss_repbox)      
        return loss