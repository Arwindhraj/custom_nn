import os
import torch
import pandas as pd
from torch import nn
from PIL import Image
from torch import save
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
  
class MyArchitecture(nn.Module):
    def __init__(self, num_classes=3):
        super(MyArchitecture, self).__init__()

        self.silu = nn.SiLU()
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

        self.conv8 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self,x):

        x1 = self.silu(self.bn1(self.conv1(x)))  # ==> 3  ==> 32
        x2 = self.silu(self.bn2(self.conv2(x1)))  # ==> 32 ==> 32
        sx2 = x1 + x2  # ==> 64
        x3 = self.silu(self.bn3(self.conv3(sx2)))  # ==> 64 ==> 64
        x4 = self.silu(self.bn4(self.conv4(x3)))  # ==> 64 ==> 64
        sx4 = x3 + x4  # ==> 128
        x5 = self.silu(self.bn5(self.conv5(sx4)))  # ==> 64 ==> 128
        x6 = self.silu(self.bn6(self.conv6(x5)))  # ==> 128 ==> 128
        sx6 = x5 + x6  # ==> 256
        x7 = self.silu(self.bn7(self.conv7(sx6)))  # ==> 128 ==> 256
        x8 = self.silu(self.bn8(self.conv8(x7)))  # ==> 256 ==> 256
        sx8 = x7 + x8  # ==> 512

        x_avg = self.global_avg_pool(sx8).view(sx8.size(0), -1)
        
        output = self.fc(x_avg)
        
        return output
    
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
    
    with open('trained_model.pt','wb') as f:
        save(model.state_dict(),f)

    print("Done!")