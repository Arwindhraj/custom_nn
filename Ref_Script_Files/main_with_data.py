import torch
from torch import nn
from torch import save 
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from torch.optim.lr_scheduler import StepLR

class CustomCOCODataset(Dataset):
    def __init__(self, annotation_file, image_root, transform=None):
        self.coco = COCO(annotation_file)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = f"{self.image_root}/{img_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]
        max_num_boxes = 4 

        while len(boxes) < max_num_boxes:
            boxes.append([0, 0, 0, 0])   
            labels.append(0)   

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if len(boxes.size()) == 1:
            boxes = boxes.unsqueeze(0)

        max_columns = max(box.size(0) for box in boxes)
        boxes = torch.stack([torch.cat([box, torch.zeros(max_columns - box.size(0), 4)], dim=0) for box in boxes])

        labels = torch.tensor(labels, dtype=torch.int64)

        if len(labels.size()) == 1:
            labels = labels.unsqueeze(1)

        max_columns = max(label.size(0) for label in labels)
        labels = torch.stack([torch.cat([label, torch.zeros(max_columns - label.size(0))], dim=0) for label in labels])

        if labels.size(1)!= boxes.size(1):
            labels = torch.cat([label, torch.zeros(labels.size(0), boxes.size(1) - label.size(1))] for label in labels)

        targets = {
            'boxes': boxes,
            'labels': labels,
        }

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'targets': targets}

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
    
if __name__ == "__main__": 

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomCOCODataset(annotation_file='Dataset/train/_annotations.coco.json', image_root='Dataset/train/', transform=transform)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    print("Done")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = MyArch().to(device)
    optimizer = SGD(clf.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    loss_function = nn.SmoothL1Loss()

    epochs = 10 
    for epoch in range(epochs):
        clf.train() 

        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to('cuda'), targets.to('cuda')

            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
