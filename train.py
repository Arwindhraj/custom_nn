import os
import torch
import pandas as pd
from PIL import Image
from torch.optim import SGD
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from components import MyArchitecture
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

class CocoFormatDataset(Dataset):
    def __init__(self,




class CustomImageDatasetExample(Dataset):
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

    training_data = CustomImageDatasetExample(annotation_file='Dataset/train/_classes.csv', img_dir='Dataset/train/', transform=transform)

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