import torch
from torch import nn, save , load
from torch.optim import SGD
from torch.utils.data import DataLoader
    
class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
