#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, get_norm
from .deform_conv_with_off import ModulatedDeformConvWithOff


class SingleHead(nn.Module):
    """
    Build single head with convolutions and coord conv.
    """
    def __init__(self, in_channel, conv_dims, num_convs, deform=False, coord=False, norm='', name=''):
        super().__init__()
        self.coord = coord
        self.conv_norm_relus = []
        if deform:
            conv_module = ModulatedDeformConvWithOff
        else:
            conv_module = Conv2d
        for k in range(num_convs):
            conv = conv_module(
                    in_channel if k==0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
            self.add_module("{}_head_{}".format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)

    def forward(self, x):
        if self.coord:
            x = self.coord_conv(x)
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x
    
    def coord_conv(self, feat):
        with torch.no_grad():
            x_pos = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
            y_pos = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
            grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
            grid_x = grid_x.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
        feat = torch.cat([feat, grid_x, grid_y], dim=1)
        return feat


class PositionHead(nn.Module):
    """
    The head used in PanopticFCN for Object Centers and Stuff Regions localization.
    """
    def __init__(self, cfg):
        super().__init__()
        thing_classes   = cfg.MODEL.POSITION_HEAD.THING.NUM_CLASSES
        stuff_classes   = cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES
        bias_value      = cfg.MODEL.POSITION_HEAD.THING.BIAS_VALUE
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.POSITION_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.POSITION_HEAD.NUM_CONVS
        deform          = cfg.MODEL.POSITION_HEAD.DEFORM
        coord           = cfg.MODEL.POSITION_HEAD.COORD
        norm            = cfg.MODEL.POSITION_HEAD.NORM

        self.position_head = SingleHead(in_channel+2 if coord else in_channel, 
                                        conv_dims, 
                                        num_convs, 
                                        deform=deform,
                                        coord=coord,
                                        norm=norm,
                                        name='position_head')
        self.out_inst = Conv2d(conv_dims, thing_classes, kernel_size=3, padding=1)
        self.out_sem = Conv2d(conv_dims, stuff_classes, kernel_size=3, padding=1)
        for layer in [self.out_inst, self.out_sem]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, bias_value)

    def forward(self, feat):
        x = self.position_head(feat)
        x_inst = self.out_inst(x)
        x_sem = self.out_sem(x)
        return x_inst, x_sem


class KernelHead(nn.Module):
    """
    The head used in PanopticFCN to generate kernel weights for both Things and Stuff.
    """
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.KERNEL_HEAD.NUM_CONVS
        deform          = cfg.MODEL.KERNEL_HEAD.DEFORM
        coord           = cfg.MODEL.KERNEL_HEAD.COORD
        norm            = cfg.MODEL.KERNEL_HEAD.NORM

        self.kernel_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims,
                                      num_convs,
                                      deform=deform,
                                      coord=coord,
                                      norm=norm,
                                      name='kernel_head')
        self.out_conv = Conv2d(conv_dims, conv_dims, kernel_size=3, padding=1)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)
       
    def forward(self, feat):
        x = self.kernel_head(feat)
        x = self.out_conv(x)
        return x


class FeatureEncoder(nn.Module):
    """
    The head used in PanopticFCN for high-resolution feature generation.
    """
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.SEMANTIC_FPN.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        num_convs       = cfg.MODEL.FEATURE_ENCODER.NUM_CONVS
        deform          = cfg.MODEL.FEATURE_ENCODER.DEFORM
        coord           = cfg.MODEL.FEATURE_ENCODER.COORD
        norm            = cfg.MODEL.FEATURE_ENCODER.NORM
        
        self.encode_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims, 
                                      num_convs, 
                                      deform=deform,
                                      coord=coord,
                                      norm=norm, 
                                      name='encode_head')

    def forward(self, feat):
        feat = self.encode_head(feat)
        return feat


class ThingGenerator(nn.Module):
    """
    The head used in PanopticFCN for Things generation with Kernel Fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels  = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        self.sim_type   = cfg.MODEL.INFERENCE.SIMILAR_TYPE
        self.sim_thres  = cfg.MODEL.INFERENCE.SIMILAR_THRES
        self.class_spec = cfg.MODEL.INFERENCE.CLASS_SPECIFIC

        self.embed_extractor = Conv2d(input_channels, conv_dims, kernel_size=1)
        for layer in [self.embed_extractor]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, feat_shape, idx_feat, idx_shape, pred_cate=None, pred_score=None):
        n, c, h, w = feat_shape
        if idx_shape>0:
            meta_weight = self.embed_extractor(idx_feat)
            meta_weight = meta_weight.reshape(*meta_weight.shape[:2], -1)
            meta_weight = meta_weight.permute(0, 2, 1)
            if not self.training:
                meta_weight, pred_cate, pred_score = self.kernel_fusion(meta_weight, pred_cate, pred_score)
            inst_pred = torch.matmul(meta_weight, x)
            inst_pred = inst_pred.reshape(n, -1, h, w)
            return inst_pred, [pred_cate, pred_score]
        else:
            return [], [None, None]

    def kernel_fusion(self, meta_weight, pred_cate, pred_score):
        meta_weight = meta_weight.squeeze(0)
        similarity = self.cal_similarity(meta_weight, meta_weight, sim_type=self.sim_type)
        label_matrix = similarity.triu(diagonal=0) >= self.sim_thres
        if self.class_spec:
            cate_matrix = pred_cate.unsqueeze(-1) == pred_cate.unsqueeze(0)
            label_matrix = label_matrix & cate_matrix
        cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2
        keep_matrix = cum_matrix.diagonal(0)
        label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(label_matrix, meta_weight) / label_norm
        pred_cate = pred_cate[keep_matrix]
        pred_score = pred_score[keep_matrix]
        return meta_weight, pred_cate, pred_score

    def cal_similarity(self, base_w, anchor_w, sim_type="cosine"):
        if sim_type == "cosine":
            a_n, b_n = base_w.norm(dim=1).unsqueeze(-1), anchor_w.norm(dim=1).unsqueeze(-1)
            a_norm = base_w / a_n.clamp(min=1e-8)
            b_norm = anchor_w / b_n.clamp(min=1e-8)
            similarity = torch.mm(a_norm, b_norm.transpose(0, 1))
        elif sim_type == "L2":
            similarity = 1. - (base_w - anchor_w).abs().clamp(min=1e-6).norm(dim=1)
        else: raise NotImplementedError
        return similarity


class StuffGenerator(nn.Module):
    """
    The head used in PanopticFCN for Stuff generation with Kernel Fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels  = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        self.conv_dims  = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        
        self.embed_extractor = Conv2d(input_channels, self.conv_dims, kernel_size=1)
        for layer in [self.embed_extractor]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, feat_shape, idx_feat, idx_mask, pred_cate=None, pred_score=None):
        n, c, h, w = feat_shape
        meta_weight = self.embed_extractor(idx_feat)
        meta_weight = meta_weight.reshape(n, -1, self.conv_dims)
        if not self.training:
            meta_weight, pred_cate, pred_score = self.kernel_fusion(meta_weight, pred_cate, pred_score)
        seg_pred = torch.matmul(meta_weight, x)
        seg_pred = seg_pred.reshape(n, -1, h, w)
        return seg_pred, [pred_cate, pred_score]

    def kernel_fusion(self, meta_weight, pred_cate, pred_score):
        unique_cate = torch.unique(pred_cate)
        meta_weight = meta_weight.squeeze(0)
        cate_matrix, uniq_matrix = pred_cate.unsqueeze(0), unique_cate.unsqueeze(1)
        label_matrix = (cate_matrix == uniq_matrix).float()
        label_norm = label_matrix.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(label_matrix, meta_weight) / label_norm
        pred_score = torch.mm(label_matrix, pred_score.unsqueeze(-1)) / label_norm
        return meta_weight, unique_cate, pred_score.squeeze(-1)

def build_position_head(cfg, input_shape=None):
    return PositionHead(cfg)

def build_kernel_head(cfg, input_shape=None):
    return KernelHead(cfg)

def build_feature_encoder(cfg, input_shape=None):
    return FeatureEncoder(cfg)

def build_thing_generator(cfg, input_shape=None):
    return ThingGenerator(cfg)

def build_stuff_generator(cfg, input_shape=None):
    return StuffGenerator(cfg)
