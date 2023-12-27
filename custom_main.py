import os
import torch
import pandas as pd
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

class MyArch(nn.Module):
    def __init__(self,num_classes=3):
        super(MyArch, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.silu1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.silu2 = nn.SiLU()

        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.silu3 = nn.SiLU()

        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.silu4 = nn.SiLU()

        self.conv5 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.silu5 = nn.SiLU()

        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.silu6 = nn.SiLU()

        self.conv7 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.silu7 = nn.SiLU()

        self.conv8 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.silu8 = nn.SiLU()

        self.conv9 = nn.Conv2d(512,1024,kernel_size=3,padding=6)
        self.bn9 = nn.BatchNorm2d(1024)
        self.silu9 = nn.SiLU()

        self.conv10 = nn.Conv2d(1024,1024,kernel_size=3,padding=6)
        self.bn10 = nn.BatchNorm2d(1024)
        self.silu10 = nn.SiLU()

        self.conv11 = nn.Conv2d(1024,num_classes,kernel_size=3)
        self.bn11 = nn.BatchNorm2d(1024)
        self.silu11 = nn.SiLU()
        
        
    def forward(self, x):

        x1 = self.silu1(self.bn1(self.conv1(x)))
        x2 = self.silu2(self.bn2(self.conv2(x1)))
        x2 += x1
        x3 = self.silu3(self.bn3(self.conv3(x2)))
        x3 += x2
        x4 = self.silu4(self.bn4(self.conv4(x3)))
        x4 += x3
        x5 = self.silu5(self.bn5(self.conv5(x4)))
        x5 += x4
        x6 = self.silu6(self.bn6(self.conv6(x5)))
        x6 += x5
        x7 = self.silu7(self.bn7(self.conv7(x6)))
        x7 += x6
        x8 = self.silu8(self.bn8(self.conv8(x7)))
        x8 += x7
        x9 = self.silu9(self.bn9(self.conv9(x8)))
        x9 += x8
        x10 = self.silu10(self.bn10(self.conv10(x9)))
        x10 += x9
        x11 = self.silu11(self.bn11(self.conv11(x10)))

        x11 = torch.mean(x11, dim=[2, 3])
        x11 = F.softmax(x11, dim=1)
        return x11
    
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
        image = read_image(img_path).float()
        label1 = self.img_labels.iloc[idx,1]
        label2 = self.img_labels.iloc[idx,2]
        label3 = self.img_labels.iloc[idx,3]

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

if __name__ == "__main__":

    training_data = CustomImageDataset(annotation_file='Dataset/train/_classes.csv', img_dir='Dataset/train/', transform=None)

    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyArch(num_classes=3).to(device)

    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Parameter type: {param.dtype}")

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    loss_function = nn.SmoothL1Loss()

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        
    print("Done!")