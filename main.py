<<<<<<< Updated upstream
import torch
from torch import nn, save , load
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms

# train = datasets.MNIST(root="data", download=True ,train=True, transform=ToTensor())
# data = DataLoader(train,32)
transform = transforms.Compose([ToTensor()])
dataset = CocoDetection(root='"D:/Project_Files/custom_nn/Dataset/train', annFile="D:/Project_Files/custom_nn/Dataset/train/_annotations.coco.json", transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

class objectdetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)), # (3,3) is the kernel size // 1 represents the color channel // 
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)# 10 outputs is mentioned as 10
            # MNIST images are originally 28x28 pixels.
            # Each convolutional layer reduces the spatial dimensions slightly (due to padding and stride).
            # Assuming a stride of 1 and no padding in your model, each convolutional layer would reduce the spatial dimensions by 2 pixels (3x3 kernel - 1).
            # After three convolutional layers, the feature maps would have a size of (28-6)*(28-6).
        )
    
    def forward(self,x):
        return self.model(x)
    
class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)  # SiLU activation function
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.transformer = TransformerBlock()  # Transformer block
        self.c3 = C3()  # C3 module with cross-convolutions

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.transformer(out)  # Apply Transformer block
        out = self.c3(out)  # Apply C3 module

        out += residual
       
        return out

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

clf = ResidualBlock().to('cuda')
optimizer = SGD(clf.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.SmoothL1Loss()

if __name__  == "__main__":
    for epoch in range(10):
        for batch in DataLoader:
            x,y = batch # x is image and y is label
            x,y = x.to('cuda'),y.to('cuda')
            predit = clf(x)
            # Pass the image to the layer and try to process and confirms with the actual label and give the loss 
            loss = loss_function(predit,y)
            # Resets the gradients of the model's parameters
            optimizer.zero_grad()
            # Backpropagates the loss through the model to compute gradients
            loss.backward()
            # Updates the model's parameters based on the gradients
            optimizer.step()

        print(f"Epochs:{epoch}")
        print(f"losses:{loss.item()}")

    with open('trained_model.pt','wb') as f:
        save(clf.state_dict(),f)
=======
import os
import torch
import pandas as pd
from torch import nn
from PIL import Image
from components import MyArchitecture
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

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
>>>>>>> Stashed changes
