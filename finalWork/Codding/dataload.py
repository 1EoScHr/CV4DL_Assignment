import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.image_filenames = data["images"]
        self.labels = data["categories"]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataLoader(batch_size=64):
    transform = transforms.Compose([   
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = CIFAR10Dataset(
        json_file='../Running/CIFAR10/annotations/cifar10_train.json',
        image_dir='../Running/CIFAR10/train_cifar10',
        transform=transform
    )

    test_dataset = CIFAR10Dataset(
        json_file='../Running/CIFAR10/annotations/cifar10_test.json',
        image_dir='../Running/CIFAR10/test_cifar10',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
