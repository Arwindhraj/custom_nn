import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = CocoDetection(root='"D:/Project_Files/custom_nn/Dataset/train', annFile="D:/Project_Files/custom_nn/Dataset/train/_annotations.coco.json", transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
