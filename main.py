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
    
clf = ImageClassification().to('cuda')
optimizer = Adam(clf.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

if __name__  == "__main__":
    for epoch in range(10):
        for batch in data:
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
