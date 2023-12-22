import torch
from torch import nn, save , load
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms

transform = transforms.Compose([ToTensor(), transforms.Resize((16, 16),antialias=True)])
dataset = CocoDetection(root='D:/Project_Files/custom_nn/Dataset/Detectron.v5i.coco/train', annFile='D:/Project_Files/custom_nn/Dataset/Detectron.v5i.coco/train/_annotations.coco.json', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

class objectdetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)), 
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        )
    
    def forward(self,x):
        return self.model(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
       
        return out

clf = ResidualBlock(in_channels=64, out_channels=64).to('cuda')
optimizer = SGD(clf.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.SmoothL1Loss()

if __name__  == "__main__":
    for epoch in range(10):
        for batch in data_loader:
            x,y = batch 
            x,y = x.to('cuda'),y.to('cuda')
            predit = clf(x)
            loss = loss_function(predit,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epochs:{epoch}")
        print(f"losses:{loss.item()}")

    with open('trained_model.pt','wb') as f:
        save(clf.state_dict(),f)
