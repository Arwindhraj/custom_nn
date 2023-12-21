import torch
from torch import nn, save , load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download=True ,train=True, transform=ToTensor())
data = DataLoader(train,32)

class ImageClassification(nn.Module):
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
    
clf = ImageClassification().to('cuda')
optimizer = Adam(clf.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

if __name__  == "__main__":
    for epoch in range(10):
        for batch in data:
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




# Residual block
import torch
import torch.nn as nn

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

        # Adding the shortcut to the output
        out += residual
       
        return out
