import torch
import numpy as np
from torch import nn, Tensor
import torchvision.ops as ops
from typing import Tuple, List
from torch.nn import functional as F


class BBox(object):

    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom)

    def tolist(self):
        return [self.left, self.top, self.right, self.bottom]

    @staticmethod
    def to_center_base(bboxes: Tensor):
        return torch.stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            (bboxes[:, 1] + bboxes[:, 3]) / 2,
            bboxes[:, 2] - bboxes[:, 0],
            bboxes[:, 3] - bboxes[:, 1]
        ], dim=1)

    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) -> Tensor:
        return torch.stack([
            center_based_bboxes[:, 0] - center_based_bboxes[:, 2] / 2,
            center_based_bboxes[:, 1] - center_based_bboxes[:, 3] / 2,
            center_based_bboxes[:, 0] + center_based_bboxes[:, 2] / 2,
            center_based_bboxes[:, 1] + center_based_bboxes[:, 3] / 2
        ], dim=1)

    @staticmethod
    def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
        transformers = torch.stack([
            (center_based_dst_bboxes[:, 0] - center_based_src_bboxes[:, 0]) / center_based_dst_bboxes[:, 2],
            (center_based_dst_bboxes[:, 1] - center_based_src_bboxes[:, 1]) / center_based_dst_bboxes[:, 3],
            torch.log(center_based_dst_bboxes[:, 2] / center_based_src_bboxes[:, 2]),
            torch.log(center_based_dst_bboxes[:, 3] / center_based_src_bboxes[:, 3])
        ], dim=1)
        return transformers

    @staticmethod
    def apply_transformer(src_bboxes: Tensor, transformers: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = torch.stack([
            transformers[:, 0] * center_based_src_bboxes[:, 2] + center_based_src_bboxes[:, 0],
            transformers[:, 1] * center_based_src_bboxes[:, 3] + center_based_src_bboxes[:, 1],
            torch.exp(transformers[:, 2]) * center_based_src_bboxes[:, 2],
            torch.exp(transformers[:, 3]) * center_based_src_bboxes[:, 3]
        ], dim=1)
        dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
        return dst_bboxes

    @staticmethod
    def iou(source: Tensor, other: Tensor) -> Tensor:
        source = source.repeat(other.shape[0], 1, 1).permute(1, 0, 2)
        other = other.repeat(source.shape[0], 1, 1)

        source_area = (source[:, :, 2] - source[:, :, 0]) * (source[:, :, 3] - source[:, :, 1])
        other_area = (other[:, :, 2] - other[:, :, 0]) * (other[:, :, 3] - other[:, :, 1])

        intersection_left = torch.max(source[:, :, 0], other[:, :, 0])
        intersection_top = torch.max(source[:, :, 1], other[:, :, 1])
        intersection_right = torch.min(source[:, :, 2], other[:, :, 2])
        intersection_bottom = torch.min(source[:, :, 3], other[:, :, 3])
        intersection_width = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area = intersection_width * intersection_height

        return intersection_area / (source_area + other_area - intersection_area)
    
    @staticmethod
    def inside(source: Tensor, other: Tensor) -> bool:
        source = source.repeat(other.shape[0], 1, 1).permute(1, 0, 2)
        other = other.repeat(source.shape[0], 1, 1)
        return ((source[:, :, 0] >= other[:, :, 0]) * (source[:, :, 1] >= other[:, :, 1]) *
                (source[:, :, 2] <= other[:, :, 2]) * (source[:, :, 3] <= other[:, :, 3]))

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        return torch.stack([
            torch.clamp(bboxes[:, 0], min=left, max=right),
            torch.clamp(bboxes[:, 1], min=top, max=bottom),
            torch.clamp(bboxes[:, 2], min=left, max=right),
            torch.clamp(bboxes[:, 3], min=top, max=bottom)
        ], dim=1)
    
class RPN(nn.Module):

    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int, int]], anchor_scales: List[int], pre_nms_top_n: int, post_nms_top_n: int):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._anchor_ratios = anchor_ratios
        self._anchor_scales = anchor_scales

        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_scales = len(self._anchor_scales)
        num_anchors = num_anchor_ratios * num_anchor_scales
  
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
 
        self._objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor]:
        
        features = self._features(features)

        objectnesses = self._objectness(features)
        transformers = self._transformer(features)

        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        
        transformers = transformers.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        return objectnesses, transformers

    def sample(self, anchor_bboxes: Tensor, gt_bboxes: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        sample_fg_indices = torch.arange(end=len(anchor_bboxes), dtype=torch.long)
        sample_selected_indices = torch.arange(end=len(anchor_bboxes), dtype=torch.long)

        anchor_bboxes = anchor_bboxes.cpu()
        gt_bboxes = gt_bboxes.cpu()

        boundary = torch.tensor(BBox(0, 0, image_width, image_height).tolist(), dtype=torch.float)
        inside_indices = BBox.inside(anchor_bboxes, boundary.unsqueeze(dim=0)).squeeze().nonzero().view(-1)

        anchor_bboxes = anchor_bboxes[inside_indices]
        sample_fg_indices = sample_fg_indices[inside_indices]
        sample_selected_indices = sample_selected_indices[inside_indices]

        labels = torch.ones(len(anchor_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(anchor_bboxes, gt_bboxes)
        anchor_max_ious, anchor_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)
        anchor_additions = (ious == gt_max_ious).nonzero()[:, 0]
        labels[anchor_max_ious < 0.3] = 0
        labels[anchor_additions] = 1
        labels[anchor_max_ious >= 0.7] = 1

        fg_indices = (labels == 1).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 - len(fg_indices)]]
        selected_indices = torch.cat([fg_indices, bg_indices])
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        gt_anchor_objectnesses = labels[selected_indices]
        gt_bboxes = gt_bboxes[anchor_assignments[fg_indices]]
        anchor_bboxes = anchor_bboxes[fg_indices]
        gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)

        gt_anchor_objectnesses = gt_anchor_objectnesses.cuda()
        gt_anchor_transformers = gt_anchor_transformers.cuda()

        sample_fg_indices = sample_fg_indices[fg_indices]
        sample_selected_indices = sample_selected_indices[selected_indices]

        return sample_fg_indices, sample_selected_indices, gt_anchor_objectnesses, gt_anchor_transformers

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor, gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor) -> Tuple[Tensor, Tensor]:
        
        cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=gt_anchor_objectnesses)
        
        smooth_l1_loss = F.smooth_l1_loss(input=anchor_transformers, target=gt_anchor_transformers, reduction='sum')
        
        smooth_l1_loss /= len(gt_anchor_transformers)

        return cross_entropy, smooth_l1_loss

    def generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int, anchor_size: int) -> Tensor:

        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]

        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]
        scales = np.array(self._anchor_scales)

        center_ys, center_xs, ratios, scales = np.meshgrid(center_ys, center_xs, ratios, scales, indexing='ij')

        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        scales = scales.reshape(-1)

        widths = anchor_size * scales * np.sqrt(1 / ratios)
        heights = anchor_size * scales * np.sqrt(ratios)

        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()

        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)

        return anchor_bboxes

    
    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) -> Tensor:

        proposal_score = objectnesses[:, 1]

        _, sorted_indices = torch.sort(proposal_score, dim=0, descending=True)
        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_bboxes = anchor_bboxes[sorted_indices]
        
        proposal_bboxes = BBox.apply_transformer(sorted_anchor_bboxes, sorted_transformers.detach())
              
        proposal_bboxes = BBox.clip(proposal_bboxes, 0, 0, image_width, image_height)
      
        keep_indices = ops.nms(proposal_bboxes, proposal_score, iou_threshold=0.7)

        proposal_bboxes = proposal_bboxes[keep_indices]
        proposal_bboxes = proposal_bboxes[:self._post_nms_top_n]

        return proposal_bboxes


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class MyArchitecture(nn.Module):
    def __init__(self, num_classes=1,anchor_ratios=[(1, 1), (1, 2), (2, 1)], anchor_scales=[1, 2, 4], pre_nms_top_n=1000, post_nms_top_n=200):
        super(MyArchitecture, self).__init__()
     
        self.silu = SiLU()
    
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        self.bn9 = nn.BatchNorm2d(512)
            
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
     
        self.fc_obj = nn.Linear(512, 1)
        
        self.fc_cls = nn.Linear(512, num_classes)
        
        self.rpn = RPN(num_features_out=512,anchor_ratios=anchor_ratios,anchor_scales=anchor_scales,pre_nms_top_n=pre_nms_top_n,post_nms_top_n=post_nms_top_n)
    
    def forward(self,x):

        x1 = self.silu(self.bn1(self.conv1(x)))  
        x2 = self.silu(self.bn2(self.conv2(x1))) 
        sx2 = x1 + x2  

        x3 = self.silu(self.bn3(self.conv3(sx2)))  
        x4 = self.silu(self.bn4(self.conv4(x3)))  

        sx4 = x3 + x4  

        x5 = self.silu(self.bn5(self.conv5(sx4)))  
        x6 = self.silu(self.bn6(self.conv6(x5)))  

        sx6 = x5 + x6  

        x7 = self.silu(self.bn7(self.conv7(sx6)))  
        x8 = self.silu(self.bn8(self.conv8(x7)))  

        sx8 = x7 + x8 

        x9 = self.silu(self.bn9(self.conv9(sx8))) 

        x_avg = self.global_avg_pool(x9).view(x9.size(0), -1)

        obj_score = torch.sigmoid(self.fc_obj(x_avg))
        cls_scores = self.fc_cls(x_avg)
        rpn_objectness, rpn_transformers = self.rpn(x9, image_width=244, image_height=244)
        
        return obj_score,cls_scores, rpn_objectness, rpn_transformers
