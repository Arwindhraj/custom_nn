from PIL import Image
from xml.etree import ElementTree as ET
from torchvision import transforms
import torch 
import os

class PotholeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.annotation_paths = self.list_paths()  # Call list_paths to initialize paths
        self.class_mapping = {'label1': 1,'label2': 0}
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        annotation_path = os.path.join(self.root_dir, self.annotation_paths[idx])

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Parse annotation
        annotation_tree = ET.parse(annotation_path)
        root = annotation_tree.getroot()
        # Extract bounding box coordinates and labels from the annotation file
        # Example: Replace this with actual parsing logic
        box_coords = []  # List of [xmin, ymin, xmax, ymax]
        labels = []  # List of corresponding labels
        for obj in root.findall('.//object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            box_coords.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_mapping[obj.find('name').text])
        # Create target dictionary
        target = {'boxes': torch.tensor(box_coords, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}

        # Apply transformations if provided
        if self.transform:
            img, target = self.transform(img, target)

        return img, target
    
    def list_paths(self):
        image_paths, annotation_paths = [], []
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(self.root_dir, split)
            for filename in os.listdir(split_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(split, filename)
                    annotation_path = os.path.join(split, filename.replace('.jpg', '.xml'))
                    image_paths.append(image_path)
                    annotation_paths.append(annotation_path)
        return image_paths, annotation_paths

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets

# Replace with actual image and annotation paths
dataset = PotholeDataset(root_dir='D:\Project_Files\Dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_sample(img, target):
    plt.imshow(img.permute(1, 2, 0))  # Change the image tensor layout

    # Adjust this part based on the structure of your target
    for box in target['boxes']:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()

# Example usage
sample_img, sample_target = dataset[0]
visualize_sample(sample_img, sample_target)
