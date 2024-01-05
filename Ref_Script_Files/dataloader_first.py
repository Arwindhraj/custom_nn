import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image

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

        # Process annotations
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'labels': labels,
        }

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'targets': targets}
    
if __name__ == '__main__':
    # Example usage
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust as needed
        transforms.ToTensor(),
    ])

    # Replace 'path/to/annotations.json' and 'path/to/images' with your actual paths
    dataset = CustomCOCODataset(annotation_file='Dataset/train/_annotations.coco.json', image_root='Dataset/train/', transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # Adjust batch size and num_workers as needed
    print("Done")
    # # Assuming you have created your DataLoader as 'dataloader'
    # for batch in dataloader:
    #     images = batch['image']
    #     targets = batch['targets']

    #     # Print the shapes to verify
    #     print(f"Image batch shape: {images.shape}")
    #     print(f"Boxes batch shape: {targets['boxes'].shape}")
    #     print(f"Labels batch shape: {targets['labels'].shape}")

    #     # Print the first image and its corresponding annotations
    #     print("First Image:")
    #     print(images[0])
    #     print("Annotations:")
    #     print("Boxes:", targets['boxes'][0])
    #     print("Labels:", targets['labels'][0])

    #     # Break the loop after printing the first batch
    #     break

