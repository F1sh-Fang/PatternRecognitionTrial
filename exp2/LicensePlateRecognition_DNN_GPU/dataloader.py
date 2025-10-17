import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 载入训练数据
class CharacterDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        image_paths = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                words = line.split()                        
                label = words[-1]
                image_path = line[:-(label.__len__()+1)]
                image_paths.append((image_path, label))     # (image_path, lable(str))
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        image_path, label = self.image_paths[index]
        image = Image.open(image_path).convert('L')
        label = int(label)
        if self.transform is not None:
            image = self.transform(image)                   # 转换为tensor格式（1，28，28）
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
