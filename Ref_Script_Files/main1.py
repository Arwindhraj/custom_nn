import torch
from torch import nn
from torch import save 
from torch.optim import SGD
from torch.utils.data import DataLoader

class MyArch(nn.Module):
    def __init__(self):
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

        self.conv10 = nn.Conv2d(1024,1024,kernel_size=3)
        self.bn10 = nn.BatchNorm2d(1024)
        self.silu10 = nn.SiLU()
        

    def forward(self, x):
        input = x
        
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
        # Gradient Flow (efficient propagation of gradients)
        return x10

clf = MyArch().to('cuda')
optimizer = SGD(clf.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.SmoothL1Loss()

epochs = 10  # You can adjust the number of epochs
for epoch in range(epochs):
    clf.train()  # Set the model in training mode

    for data in train_loader:
        inputs, targets = data
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        optimizer.zero_grad()
        outputs = clf(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
