#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
from functools import partial

def topk_score(scores, K=40, score_shape=None):
    """
    get top K point in score map
    """
    batch, channel, height, width = score_shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    # div by K because index is grouped by K(C x K shape)
    topk_clses = index // K
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
