import numpy as np
import torch
import torch.nn.functional as F

from .utils import multi_apply


class GenerateGT(object):
    """
    Generate ground truth for Panoptic FCN.
    """
    def __init__(self, cfg):
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.ignore_val = cfg.MODEL.IGNORE_VALUE
        self.tensor_dim = cfg.MODEL.TENSOR_DIM
        self.common_stride = cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE
        self.thing_classes = cfg.MODEL.POSITION_HEAD.THING.NUM_CLASSES
        self.stuff_classes = cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES
        self.sem_with_thing = cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING
        self.min_overlap = cfg.MODEL.POSITION_HEAD.THING.MIN_OVERLAP
        self.instance_scales = cfg.MODEL.KERNEL_HEAD.INSTANCE_SCALES
        self.gaussian_sigma = cfg.MODEL.POSITION_HEAD.THING.GAUSSIAN_SIGMA
        self.center_type = cfg.MODEL.POSITION_HEAD.THING.CENTER_TYPE
        
    def assign_multi_scale(self, gt_instances, gt_semantics, image_shape, inst_outsize, feat_shape):
        """
        Assign ground truth to different FPN stages.
        """
        assert len(feat_shape) == len(self.instance_scales)
        gt_center_labels, gt_inst_labels, gt_index_labels = [], [], []
        index_mask_labels, gt_class_labels = [], []
        gt_sem_scores, gt_sem_labels, gt_sem_masks, index_masks_sem = [], [], [], []

        # gt for stuff
        gt_sem = gt_semantics.to(dtype=torch.int64, device=self.device)
        if self.sem_with_thing:
            gt_sem[gt_sem==self.ignore_val] = self.stuff_classes
        else:
            gt_sem[gt_sem==self.ignore_val] = 0
            gt_sem -= 1
            gt_sem[gt_sem==-1] = self.stuff_classes
        gt_sem = F.one_hot(gt_sem, num_classes=self.stuff_classes+1)
        gt_sem = gt_sem.permute(2, 0, 1).float()
        gt_sem = gt_sem[:self.stuff_classes, ...]
        h, w = gt_sem.shape[-2:]
        new_h, new_w = int(h / self.common_stride + 0.5), int(w / self.common_stride + 0.5)
        gt_sem = F.interpolate(gt_sem.unsqueeze(0), size=(new_h, new_w), 
                               mode='bilinear', align_corners=False)[0]
        gt_semantics = torch.zeros(self.stuff_classes, *inst_outsize, device=self.device)
        gt_semantics[:, :new_h, :new_w] = gt_sem
        if hasattr(gt_instances, 'gt_masks'):
            # gt for things
            use_mass = ('mass' in self.center_type)
            bit_masks, mass_centers, scales = GenerateGT.decode_bitwise_mask(common_stride=self.common_stride,
                                                                             gt_masks=gt_instances.gt_masks,
                                                                             device=self.device,
                                                                             use_mass=use_mass)
            gt_boxes_raw = gt_instances.gt_boxes
            gt_class_raw = gt_instances.gt_classes
            gt_areas_raw = torch.sqrt(gt_boxes_raw.area())

            if use_mass:
                gt_centers_raw = mass_centers
            else:
                gt_centers_raw = gt_boxes_raw.get_centers()
                gt_centers_raw = gt_centers_raw / self.common_stride
            for layer_idx in range(len(feat_shape)):
                gt_center_per, gt_inst_per, gt_index_per, index_mask_per, classes_per = \
                            self._label_assignment(gt_class_raw, bit_masks, gt_centers_raw, 
                                                   gt_boxes_raw, gt_areas_raw,
                                                   self.instance_scales[layer_idx],
                                                   image_shape, inst_outsize, 
                                                   feat_shape[layer_idx])
                gt_semscore_per, gt_semseg_per, sem_mask_per, index_mask_sem_per  = \
                            self._sem_assign(gt_semantics, feat_shape[layer_idx])
                gt_center_labels.append(gt_center_per)
                gt_inst_labels.append(gt_inst_per)
                gt_index_labels.append(gt_index_per)
                index_mask_labels.append(index_mask_per)
                gt_class_labels.append(classes_per)
                gt_sem_scores.append(gt_semscore_per)
                gt_sem_labels.append(gt_semseg_per)
                gt_sem_masks.append(sem_mask_per)
                index_masks_sem.append(index_mask_sem_per)
        else:
            for layer_idx in range(len(feat_shape)):
                gt_center_per = torch.zeros(self.thing_classes, *feat_shape[layer_idx], device=self.device)
                gt_inst_per = self.ignore_val * torch.ones(self.tensor_dim, *inst_outsize, device=self.device)
                gt_index_per = torch.zeros(self.tensor_dim, device=self.device)
                index_mask_per = torch.zeros(self.tensor_dim, device=self.device)
                classes_per = torch.zeros(self.tensor_dim, device=self.device)
                gt_semscore_per, gt_semseg_per, sem_mask_per, index_mask_sem_per = \
                        self._sem_assign(gt_semantics, feat_shape[layer_idx])
                gt_center_labels.append(gt_center_per)
                gt_inst_labels.append(gt_inst_per)
                gt_index_labels.append(gt_index_per)
                index_mask_labels.append(index_mask_per)
                gt_class_labels.append(classes_per)
                gt_sem_scores.append(gt_semscore_per)
                gt_sem_labels.append(gt_semseg_per)
                gt_sem_masks.append(sem_mask_per)
                index_masks_sem.append(index_mask_sem_per)

        return gt_center_labels, gt_inst_labels, gt_index_labels, index_mask_labels, gt_class_labels, gt_sem_scores, gt_sem_labels, gt_sem_masks, index_masks_sem

    def _label_assignment(self, gt_class_raw, gt_masks_raw, gt_centers_raw, gt_boxes_raw, gt_areas_raw, inst_scale, image_shape, inst_outsize, feat_shape):
        """
        Assign labels for Things to each FPN stage.
        """
        
        # init gt tensors
        gt_scoremap = torch.zeros(self.thing_classes, *feat_shape, device=self.device)
        gt_instance = self.ignore_val * torch.ones(self.tensor_dim, *inst_outsize, device=self.device)
        gt_index = torch.zeros(self.tensor_dim, device=self.device)
        inst_mask = torch.zeros(self.tensor_dim, device=self.device)
        gt_class = torch.zeros(self.tensor_dim, device=self.device)
        box_rescale = [feat_shape[-2]/image_shape[-2], feat_shape[-1]/image_shape[-1]]
        center_rescale = [feat_shape[-2]/inst_outsize[-2], feat_shape[-1]/inst_outsize[-1]]
        gt_assign_mask = ((gt_areas_raw >= inst_scale[0]) & (gt_areas_raw <= inst_scale[1]))

        if gt_assign_mask.sum() == 0:
            return gt_scoremap, gt_instance, gt_index, inst_mask, gt_class

        # assign center
        centers = gt_centers_raw[gt_assign_mask]
        centers[...,0] *= center_rescale[1]
        centers[...,1] *= center_rescale[0]
        centers_int = centers.to(torch.int64)
        centers_int[:,0].clamp_(min=0, max=feat_shape[1])
        centers_int[:,1].clamp_(min=0, max=feat_shape[0])
        # assign masks
        bit_masks = gt_masks_raw[gt_assign_mask]
        num_inst = len(bit_masks)
        gt_instance[:num_inst] = 0.
        gt_instance[:num_inst, :bit_masks.shape[1], :bit_masks.shape[2]] = bit_masks
        gt_index[:num_inst] = centers_int[..., 1] * feat_shape[1] + centers_int[..., 0]
        inst_mask[:num_inst] = 1
        # assign classes
        classes = gt_class_raw[gt_assign_mask]
        gt_class[:num_inst] = classes
        # assign score map
        box_tensor = gt_boxes_raw[gt_assign_mask].tensor
        wh = torch.zeros_like(centers)
        wh[..., 0] = (box_tensor[..., 2] - box_tensor[..., 0]) * box_rescale[1]
        wh[..., 1] = (box_tensor[..., 3] - box_tensor[..., 1]) * box_rescale[0]
        GenerateGT.generate_score_map(gt_scoremap, classes, wh,
                                      centers_int, self.min_overlap,
                                      sigma_factor=self.gaussian_sigma,
                                      device=self.device)
        
        return gt_scoremap, gt_instance, gt_index, inst_mask, gt_class

    def _sem_assign(self, gt_semantic, feat_shape): 
        """
        Assign labels for Stuff to each FPN stage.
        """
        inst_mask = torch.zeros(self.stuff_classes, device=self.device)
        gt_sem_labels = torch.zeros_like(gt_semantic)
        gt_scoremap = F.interpolate(gt_semantic.unsqueeze(0), size=feat_shape, mode='bilinear', align_corners=False)[0]
        gt_scoremap = gt_scoremap.clamp(max=1.0)
        gt_scoremap[gt_scoremap<0.5] = 0.0
        gt_assign_mask = gt_scoremap.reshape(self.stuff_classes, -1).sum(dim=-1) > 0
        num_sem = gt_assign_mask.sum()
        gt_sem_labels[:num_sem] = gt_semantic[gt_assign_mask]
        gt_sem_masks = torch.zeros_like(gt_scoremap)
        gt_sem_masks[:num_sem] = gt_scoremap[gt_assign_mask].bool().float()
        inst_mask[:num_sem] = 1
        return gt_scoremap, gt_sem_labels, gt_sem_masks, inst_mask

    @torch.no_grad()
    def generate(self, batched_input, images, features, inst_feat, test_only=False):
        """
        Generate ground truth of multi-stages according to the input.
        """
        try:
            gt_instances = [x["instances"] for x in batched_input]
            with_object = True
        except:
            gt_instances = [{}]
            with_object = False

        gt_semantics = [x["sem_seg"] for x in batched_input]
        feat_shape = [x.shape[-2:] for x in features]
        inst_shape = inst_feat.shape[-2:]
        image_shape = images.tensor.shape[-2:]

        gt_center_labels, gt_inst_labels, gt_index_labels, index_mask_labels, gt_class_labels, \
                gt_sem_scores, gt_sem_labels, gt_sem_masks, index_masks_sem = \
                multi_apply(self.assign_multi_scale, gt_instances, gt_semantics,
                            image_shape=image_shape, inst_outsize=inst_shape, 
                            feat_shape=feat_shape)

        gt_centers, gt_insts, gt_indexes, index_masks, gt_classes = [[] for _ in range(5)]
        gt_scores_sem, gt_labels_sem, gt_mask_sem, gt_index_sem = [[] for _ in range(4)]
        for _idx in range(len(feat_shape)):
            _sem_scores = [x[_idx] for x in gt_sem_scores]
            _sem_labels = [x[_idx] for x in gt_sem_labels]
            _masks_sem = [x[_idx] for x in gt_sem_masks]
            _indexes_sem = [x[_idx] for x in index_masks_sem]
            
            gt_scores_sem.append(torch.stack(_sem_scores, dim=0))
            gt_labels_sem.append(torch.stack(_sem_labels, dim=0))
            gt_mask_sem.append(torch.stack(_masks_sem, dim=0))
            gt_index_sem.append(torch.stack(_indexes_sem, dim=0))
            if with_object:
                _centers = [x[_idx] for x in gt_center_labels]
                _insts = [x[_idx] for x in gt_inst_labels]
                _indexes = [x[_idx] for x in gt_index_labels]
                _masks = [x[_idx] for x in index_mask_labels]
                _classes = [x[_idx] for x in gt_class_labels]
                gt_centers.append(torch.stack(_centers, dim=0))
                gt_insts.append(torch.stack(_insts, dim=0))
                gt_indexes.append(torch.stack(_indexes, dim=0))
                index_masks.append(torch.stack(_masks, dim=0))
                gt_classes.append(torch.stack(_classes, dim=0))
        
        gt_dict = {
            "center": gt_centers,
            "inst": gt_insts,
            "index": gt_indexes,
            "index_mask": index_masks,
            "class": gt_classes,
            "sem_scores": gt_scores_sem,
            "sem_labels":gt_labels_sem,
            "sem_masks":gt_mask_sem,
            "sem_index": gt_index_sem,
        }
        return gt_dict

    @staticmethod
    def decode_bitwise_mask(common_stride, gt_masks, device, use_mass=False):
        """
        Decode bitmask for Things and calculate mass centers.
        """
        bit_masks = gt_masks.tensor
        bit_masks = bit_masks.to(device)
        mass_centers = []
        scale = 1. / common_stride
        h, w = bit_masks.shape[-2:]
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        bit_masks = F.interpolate(bit_masks.float().unsqueeze(0), size=(new_h, new_w),
                                  mode="bilinear", align_corners=False)[0]
        # center type [x, y]
        if use_mass:
            for _mask in bit_masks:
                mass_center = torch.nonzero(_mask.float(), as_tuple=False)
                mass_center = mass_center.float().mean(dim=0)
                mass_centers.append(mass_center[[1,0]])
            mass_centers = torch.stack(mass_centers)
        return bit_masks, mass_centers, [new_h/h, new_w/w]

    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap, sigma_factor=6, device=None):
        """
        Generate gaussian-based score map for Things in each stage.
        """
        radius = GenerateGT.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            GenerateGT.draw_gaussian(fmap[channel_index], centers_int[i], radius[i],
                                     sigma_factor=sigma_factor, device=device)

    @staticmethod
    def get_gaussian_radius(box_tensor, min_overlap):
        """
        Calculate Gaussian radius based on box size.
        This algorithm is copyed from CornerNet.
        box_tensor (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        """
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1, sigma_factor=6, device=None):
        """
        Draw gaussian-based score map.
        """
        diameter = 2 * radius + 1
        gaussian = GenerateGT.gaussian2D((radius, radius), sigma=diameter / sigma_factor)
        gaussian = torch.Tensor(gaussian).to(device=device)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
