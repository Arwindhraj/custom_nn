import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import save
from torch.optim import SGD
from torch import nn, Tensor
import torchvision.ops as ops
from typing import Tuple, List
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

class SiLU(nn.Module):
    # SiLU activation https://arxiv.org/pdf/1606.08415.pdf
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# Spatial Dimensions are not changed 
class MyArchitecture(nn.Module):
    def __init__(self, num_classes=1,anchor_ratios=[(1, 1), (1, 2), (2, 1)], anchor_scales=[1, 2, 4], pre_nms_top_n=1000, post_nms_top_n=200):
        super(MyArchitecture, self).__init__()

        # Activation Function
        self.silu = SiLU()

        # Convolution Layers
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
        
        # Average Pooling Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Predicting objectness score 
        self.fc_obj = nn.Linear(512, 1)
        # Class scores
        self.fc_cls = nn.Linear(512, num_classes)


        self.rpn = RPN(num_features_out=512,anchor_ratios=anchor_ratios,anchor_scales=anchor_scales,pre_nms_top_n=pre_nms_top_n,post_nms_top_n=post_nms_top_n)
    
    def forward(self,x):

        x1 = self.silu(self.bn1(self.conv1(x)))  # ==> 3  ==> 32
        x2 = self.silu(self.bn2(self.conv2(x1))) # ==> 32 ==> 32
        sx2 = x1 + x2  # ==> 64

        x3 = self.silu(self.bn3(self.conv3(sx2)))  # ==> 64 ==> 64
        x4 = self.silu(self.bn4(self.conv4(x3)))  # ==> 64 ==> 64

        sx4 = x3 + x4  # ==> 128

        x5 = self.silu(self.bn5(self.conv5(sx4)))  # ==> 64 ==> 128
        x6 = self.silu(self.bn6(self.conv6(x5)))  # ==> 128 ==> 128

        sx6 = x5 + x6  # ==> 256

        x7 = self.silu(self.bn7(self.conv7(sx6)))  # ==> 128 ==> 256
        x8 = self.silu(self.bn8(self.conv8(x7)))  # ==> 256 ==> 256

        sx8 = x7 + x8 # ==> 512

        x9 = self.silu(self.bn9(self.conv9(sx8))) # ==> 256 ==> 512

        x_avg = self.global_avg_pool(x9).view(x9.size(0), -1)

        obj_score = torch.sigmoid(self.fc_obj(x_avg))
        cls_scores = self.fc_cls(x_avg)
        rpn_objectness, rpn_transformers = self.rpn(x9, image_width=244, image_height=244)
        
        return obj_score,cls_scores, rpn_objectness, rpn_transformers
    
# For RPN
class BBox(object):

    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        # Coordinates of the bounding box
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    # Return String representation of the Bounding box
    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom)

    # Return List representation of the Bounding box
    def tolist(self):
        return [self.left, self.top, self.right, self.bottom]

    # Converts Bounding boxes from corner based to centre based and return Tensor[x,y,w,h]
    @staticmethod
    def to_center_base(bboxes: Tensor):
        return torch.stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            (bboxes[:, 1] + bboxes[:, 3]) / 2,
            bboxes[:, 2] - bboxes[:, 0],
            bboxes[:, 3] - bboxes[:, 1]
        ], dim=1)

    # Converts Bounding boxes from centre based to corner based and return Tensor[left,top,right,bottom]
    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) -> Tensor:
        return torch.stack([
            center_based_bboxes[:, 0] - center_based_bboxes[:, 2] / 2,
            center_based_bboxes[:, 1] - center_based_bboxes[:, 3] / 2,
            center_based_bboxes[:, 0] + center_based_bboxes[:, 2] / 2,
            center_based_bboxes[:, 1] + center_based_bboxes[:, 3] / 2
        ], dim=1)

    # Calculates the transformation required to transform source bounding boxes to destination bounding boxes
    # Adjusting the anchor boxes 
    # Return transformation parameters in Tensor[dx, dy, d(log(width)), d(log(height))]
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

    # Apply set of transformations to source bboxes and return transformed bboxes in Tensor 
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

    # Calculate Intersection over Union and return Tensor for pair of bounding boxes
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

    # Checks if the source bounding boxes are inside the other bounding boxes and returns Boolean Tensors 
    @staticmethod
    def inside(source: Tensor, other: Tensor) -> bool:
        source = source.repeat(other.shape[0], 1, 1).permute(1, 0, 2)
        other = other.repeat(source.shape[0], 1, 1)
        return ((source[:, :, 0] >= other[:, :, 0]) * (source[:, :, 1] >= other[:, :, 1]) *
                (source[:, :, 2] <= other[:, :, 2]) * (source[:, :, 3] <= other[:, :, 3]))

    # Clips the bounding boxes to be within a specified region and return Tensor[left,top,right,bottom]
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

        # Stores the anchor ratios and scales provided as arguments to the constructor
        self._anchor_ratios = anchor_ratios
        self._anchor_scales = anchor_scales

        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_scales = len(self._anchor_scales)
        num_anchors = num_anchor_ratios * num_anchor_scales

        # Store the number of top anchors to consider before and after Non-Maximum Suppression (NMS)
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n


        # Predicting objectness scores and transformer parameters
        self._objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor]:
        # Pass the features to the Conv2d 
        features = self._features(features)

        # Computes the objectness scores and transformer predictions using the output of features
        objectnesses = self._objectness(features)
        transformers = self._transformer(features)

        # Permutes the dimensions of the tensors for convenience in subsequent operations
        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        # Reshapes the tensors into 2D format for easier handling
        transformers = transformers.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        return objectnesses, transformers

    def sample(self, anchor_bboxes: Tensor, gt_bboxes: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # Indices for positive (foreground) and all (selected) samples.
        sample_fg_indices = torch.arange(end=len(anchor_bboxes), dtype=torch.long)
        sample_selected_indices = torch.arange(end=len(anchor_bboxes), dtype=torch.long)

        # Converts tensors to CPU and removes anchor boxes that are outside the image boundaries.
        anchor_bboxes = anchor_bboxes.cpu()
        gt_bboxes = gt_bboxes.cpu()

        # remove cross-boundary
        boundary = torch.tensor(BBox(0, 0, image_width, image_height).tolist(), dtype=torch.float)
        inside_indices = BBox.inside(anchor_bboxes, boundary.unsqueeze(dim=0)).squeeze().nonzero().view(-1)

        anchor_bboxes = anchor_bboxes[inside_indices]
        sample_fg_indices = sample_fg_indices[inside_indices]
        sample_selected_indices = sample_selected_indices[inside_indices]

        # Find labels for each `anchor_bboxes` and Computes IoU between anchor boxes and ground truth boxes to assign labels
        labels = torch.ones(len(anchor_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(anchor_bboxes, gt_bboxes)
        anchor_max_ious, anchor_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)
        anchor_additions = (ious == gt_max_ious).nonzero()[:, 0]
        labels[anchor_max_ious < 0.3] = 0
        labels[anchor_additions] = 1
        labels[anchor_max_ious >= 0.7] = 1

        # select 256 samples
        # Selects positive (foreground) and negative (background) samples for training.
        # Randomly samples 128 positive samples and fills the remaining slots with negative samples.
        fg_indices = (labels == 1).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 - len(fg_indices)]]
        selected_indices = torch.cat([fg_indices, bg_indices])
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        # Gathers ground truth labels, ground truth bounding boxes, anchor boxes for positive samples,
        # and calculates corresponding transformer values.
        gt_anchor_objectnesses = labels[selected_indices]
        gt_bboxes = gt_bboxes[anchor_assignments[fg_indices]]
        anchor_bboxes = anchor_bboxes[fg_indices]
        gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)

        # Converts the tensors to CUDA (GPU)
        gt_anchor_objectnesses = gt_anchor_objectnesses.cuda()
        gt_anchor_transformers = gt_anchor_transformers.cuda()

        # Returns indices for positive samples, indices for all selected samples, ground truth objectness labels, 
        # and ground truth transformer values for positive samples.
        sample_fg_indices = sample_fg_indices[fg_indices]
        sample_selected_indices = sample_selected_indices[selected_indices]

        return sample_fg_indices, sample_selected_indices, gt_anchor_objectnesses, gt_anchor_transformers

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor, gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor) -> Tuple[Tensor, Tensor]:
        # Cross entropy loss between predicted objectness scores and ground truth objectness labels 
        cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=gt_anchor_objectnesses)

        # Smooth L1 loss between predicted bounding box transformations (anchor_transformers) and ground truth bounding box transformations 
        smooth_l1_loss = F.smooth_l1_loss(input=anchor_transformers, target=gt_anchor_transformers, reduction='sum')
        # Normalized by dividing by the number of positive samples
        smooth_l1_loss /= len(gt_anchor_transformers)

        return cross_entropy, smooth_l1_loss

    def generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int, anchor_size: int) -> Tensor:

        # Computes evenly spaced anchor centers along the height and width of the image.
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]

        # Extracts anchor ratios and scales from the RPN configuration.
        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]
        scales = np.array(self._anchor_scales)

        # Creates a 4D grid with dimensions (#center_ys, #center_xs, #ratios, #scales) using meshgrid.
        center_ys, center_xs, ratios, scales = np.meshgrid(center_ys, center_xs, ratios, scales, indexing='ij')

        # Reshapes the parameters 
        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        scales = scales.reshape(-1)

        # Calculates anchor widths and heights
        widths = anchor_size * scales * np.sqrt(1 / ratios)
        heights = anchor_size * scales * np.sqrt(ratios)

        # Creates anchor boxes in a center-based representation
        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()

        # Converts the center-based anchor boxes to regular anchor boxes
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)

        return anchor_bboxes

    # Generating final region proposals based on the predicted objectness scores and bounding box transformations
    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) -> Tensor:

        # Objectness scores for foreground 
        proposal_score = objectnesses[:, 1]

        # Sorts the proposals based on their objectness scores in descending order.
        _, sorted_indices = torch.sort(proposal_score, dim=0, descending=True)
        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_bboxes = anchor_bboxes[sorted_indices]
        
        # Applies the bounding box transformations to the sorted anchor boxes, producing the final proposal boxes.
        proposal_bboxes = BBox.apply_transformer(sorted_anchor_bboxes, sorted_transformers.detach())
        
        # Clips the proposal boxes to ensure they stay within the image boundaries.
        proposal_bboxes = BBox.clip(proposal_bboxes, 0, 0, image_width, image_height)

        # Use PyTorch's NMS
        keep_indices = ops.nms(proposal_bboxes, proposal_score, iou_threshold=0.7)

        proposal_bboxes = proposal_bboxes[keep_indices]
        proposal_bboxes = proposal_bboxes[:self._post_nms_top_n]

        return proposal_bboxes
    
class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image = Image.open(img_path).convert("RGB")
        label1 = self.img_labels.iloc[idx,1]
        label2 = self.img_labels.iloc[idx,2]
        label3 = self.img_labels.iloc[idx,3]
        # remove the above (label1 and label2 and label3) and replace them with [x,y,w,h] coordinates

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label1 = self.target_transform(label1)
            label2 = self.target_transform(label2)
            label3 = self.target_transform(label3)
        return image, label1, label2, label3
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, label1, label2, label3) in enumerate(dataloader):
        y = torch.stack([label1, label2, label3], dim=1)
        images, y = images.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    input_size = (224, 224)

    transform = transforms.Compose([transforms.Resize(input_size),transforms.ToTensor(),])

    training_data = CustomImageDataset(annotation_file='Dataset/train/_classes.csv', img_dir='Dataset/train/', transform=transform)

    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyArchitecture(num_classes=3).to(device)

    print(f"Number of parameters in the model: {count_parameters(model)}")

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Model Parameters: ", model.parameters())
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    loss_function = nn.SmoothL1Loss()

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
    
    model_save_path = 'trained_model.pt'
    torch.save(model.state_dict(), model_save_path)

    print("Done!")