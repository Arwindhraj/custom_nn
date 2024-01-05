from torch import nn

class MyArchitecture(nn.Module):
    def __init__(self):
        super(MyArchitecture, self).__init__()

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
    
    def forward(self,x):

        x1 = nn.SiLU(self.bn1(self.conv1(x))) # ==> 3  ==> 32
        x2 = nn.SiLU(self.bn2(self.conv2(x1))) # ==> 32 ==> 32
        sx2 = x1 + x2 # ==> 64
        x3 = nn.SiLU(self.bn3(self.conv3(x2))) # ==> 32 ==> 64
        x4 = nn.SiLU(self.bn4(self.conv4(sx2))) # ==> 64 ==> 64
        sx4 = x3 + x4 # ==> 128
        x5 = nn.SiLU(self.bn5(self.conv5(x4))) # ==> 64 ==> 128
        x6 = nn.SiLU(self.bn6(self.conv6(sx4))) # ==> 128 ==> 128
        sx6 = x5 + x6 # ==> 256
        x7 = nn.SiLU(self.bn7(self.conv7(x6))) # ==> 128 ==> 256
        x8 = nn.SiLU(self.bn8(self.conv8(sx6))) # ==> 256 ==> 256
        sx8 = x7 + x8
        
        return sx8