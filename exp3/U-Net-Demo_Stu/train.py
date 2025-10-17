import os
import sys
import random
import cv2
import warnings
import json
import datetime
from tqdm import tqdm
from skimage import io
import skimage
from skimage.draw import polygon
from skimage.transform import resize
from skimage.io import imread, imshow, imread_collection, concatenate_images
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from visualize import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics import JaccardIndex  # 用于计算IoU
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


class BalloonDataset(Dataset):
    def __init__(self, annotations, dataset_dir, img_size=(128, 128), transform=None):
        self.annotations = annotations
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        tid = list(self.annotations.keys())[idx]
        a = self.annotations[tid]
        mask, image, _, _, _, _ = get_mask(a, self.dataset_dir)
        # cv2.imwrite("image.jpg", image)
        # cv2.imwrite("mask.jpg", mask*255)
        # print(mask.shape)
        # print(image.shape)
        # cv2.imshow("image", image)
        # print("mask:",np.max(mask))
        # Resize and normalize
        mask = resize(mask, self.img_size, mode='constant', preserve_range=True).astype(np.float32)
        image = resize(image, self.img_size, mode='constant', preserve_range=True).astype(np.float32) / 255.0
        # print(mask.shape)
        #
        # print(image.shape)
        # print("\n")
        # Convert to tensor and add channels
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Single channel mask
        # image = torch.tensor(image, dtype=torch.float32)  # Channels-first for PyTorch
        image = image.clone().detach().requires_grad_(True)

        # print(image.shape)
        return image, mask

# def get_mask(a, dataset_dir):
#     image_path = os.path.join(dataset_dir, a['filename'])
#     image = io.imread(image_path)
#     height, width = image.shape[:2]
#     polygons = [r['shape_attributes'] for r in a['regions'].values()]
#     mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
#
#     for i, p in enumerate(polygons):
#         rr, cc = polygon(p['all_points_y'], p['all_points_x'])
#         rr = list(map(lambda x: height - 1 if x > height - 1 else x, rr))
#         cc = list(map(lambda x: width - 1 if x > width - 1 else x, cc))
#         mask[rr, cc, i] = 1
#
#     mask = mask.astype(bool)
#     return mask, image, height, width, None, None


def get_mask(a, dataset_dir):
    image_path = os.path.join(dataset_dir, a['filename'])
    image = io.imread(image_path)
    height, width = image.shape[:2]
    polygons = [r['shape_attributes'] for r in a['regions'].values()]
    mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

    for i, p in enumerate(polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        # print(max(cc))
        rr = list(map(lambda x: height-1 if x > height-1 else x, rr))
        cc = list(map(lambda x: width-1 if x > width-1 else x, cc))
        # print("i:",i)
        mask[rr, cc, i] = 1

    mask, class_ids = mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    # boxes = extract_bboxes(mask)
    boxes = extract_bboxes(resize(mask, (128, 128), mode='constant', preserve_range=True))

    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]

    class_id = top_ids[0]
    # Pull masks of instances belonging to the same class.
    m = mask[:, :, np.where(class_ids == class_id)[0]]
    m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)

    return m, image, height, width, class_ids, boxes

### 加载数据集
annotations_path = "dataset/balloon/train_fake/via_region_data.json"
dataset_dir = 'dataset/balloon/train_fake'
annotations = json.load(open(annotations_path))
train_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = BalloonDataset(annotations, dataset_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# 验证数据集
annotations_test_path = "dataset/balloon/val/via_region_data.json"
testset_dir = 'dataset/balloon/val'
annotations_test = json.load(open(annotations_test_path))
test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)

class UNetModel(nn.Module):
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, out_channels=1):
        super(UNetModel, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=IMG_CHANNELS, out_channels=8, kernel_size=3, stride=1, padding=1)  # 由128*128*1变成了128*128*8
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)  # 由128*128*8变成了128*128*8
        self.relu1_2 = nn.ReLU(inplace=True)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样，图片大小减半，通道数不变，由128*128*8变成64*64*8

        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1,padding=1)  # 64*64*8->64*64*16
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  # 64*64*16->64*64*16
        self.relu2_2 = nn.ReLU(inplace=True)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样  64*64*16->32*32*16

        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1)  # 32*32*16->32*32*32
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 32*32*32->32*32*32
        self.relu3_2 = nn.ReLU(inplace=True)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样  32*32*32->16*16*32

        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1)  # 16*16*32->16*16*64
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 16*16*64->16*16*64
        self.relu4_2 = nn.ReLU(inplace=True)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 采用最大池化进行下采样 16*16*64->8*8*64

        self.conv5_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1)  # 8*8*64->8*8*128
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 8*8*128->8*8*128
        self.relu5_2 = nn.ReLU(inplace=True)

        # 接下来实现上采样中的up-conv2*2
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2,
                                            padding=0)  # 28*28*1024->56*56*512

        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1,
                                 padding=0)  # 56*56*1024->54*54*512
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)  # 54*54*512->52*52*512
        self.relu6_2 = nn.ReLU(inplace=True)

        self.up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2,
                                            padding=0)  # 52*52*512->104*104*256

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,
                                 padding=0)  # 104*104*512->102*102*256
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  # 102*102*256->100*100*256
        self.relu7_2 = nn.ReLU(inplace=True)

        self.up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2,
                                            padding=0)  # 100*100*256->200*200*128

        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                                 padding=0)  # 200*200*256->198*198*128
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  # 198*198*128->196*196*128
        self.relu8_2 = nn.ReLU(inplace=True)

        self.up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2,
                                            padding=0)  # 196*196*128->392*392*64

        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                                 padding=0)  # 392*392*128->390*390*64
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)  # 390*390*64->388*388*64
        self.relu9_2 = nn.ReLU(inplace=True)

        # 最后的conv1*1
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    # 中心裁剪，
    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        # 如果原始张量的尺寸为10，而delta为2，那么"delta:tensor_size - delta"将截取从索引2到索引8的部分，长度为6，以使得截取后的张量尺寸变为6。
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)  # 这个后续需要使用
        down1 = self.maxpool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        x4 = self.relu2_2(x4)  # 这个后续需要使用
        down2 = self.maxpool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        x6 = self.relu3_2(x6)  # 这个后续需要使用
        down3 = self.maxpool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        x8 = self.relu4_2(x8)  # 这个后续需要使用
        down4 = self.maxpool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        x10 = self.relu5_2(x10)

        # 第一次上采样，需要"Copy and crop"（复制并裁剪）
        up1 = self.up_conv_1(x10)  # 得到56*56*512
        # 需要对x8进行裁剪，从中心往外裁剪
        crop1 = self.crop_tensor(x8, up1)
        up_1 = torch.cat([crop1, up1], dim=1)

        y1 = self.conv6_1(up_1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.relu6_2(y2)

        # 第二次上采样，需要"Copy and crop"（复制并裁剪）
        up2 = self.up_conv_2(y2)
        # 需要对x6进行裁剪，从中心往外裁剪
        crop2 = self.crop_tensor(x6, up2)
        up_2 = torch.cat([crop2, up2], dim=1)

        y3 = self.conv7_1(up_2)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.relu7_2(y4)

        # 第三次上采样，需要"Copy and crop"（复制并裁剪）
        up3 = self.up_conv_3(y4)
        # 需要对x4进行裁剪，从中心往外裁剪
        crop3 = self.crop_tensor(x4, up3)
        up_3 = torch.cat([crop3, up3], dim=1)

        y5 = self.conv8_1(up_3)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.relu8_2(y6)

        # 第四次上采样，需要"Copy and crop"（复制并裁剪）
        up4 = self.up_conv_4(y6)
        # 需要对x2进行裁剪，从中心往外裁剪
        crop4 = self.crop_tensor(x2, up4)
        up_4 = torch.cat([crop4, up4], dim=1)

        y7 = self.conv9_1(up_4)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.relu9_2(y8)

        # 最后的conv1*1
        out = self.conv_10(y8)
        return out



model = UNetModel(128, 128, 3).to(device)
criterion = nn.BCEWithLogitsLoss() # nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 计算IoU的函数
def compute_iou(pred, target, threshold=0.5):
    """
    计算 IoU (Intersection over Union) 值
    Args:
        pred (Tensor): 预测值，尺寸为 (batch_size, 1, H, W)
        target (Tensor): 真实标签，尺寸为 (batch_size, 1, H, W)
        threshold (float): 用于二值化预测结果的阈值，默认0.5
    Returns:
        iou (float): IoU值
    """
    pred = torch.sigmoid(pred)  # 转化为0-1之间
    pred = (pred > threshold).float()  # 二值化预测
    intersection = torch.sum(pred * target)  # 交集
    union = torch.sum(pred) + torch.sum(target) - intersection  # 并集
    iou = intersection / (union + 1e-6)  # 避免除零错误
    return iou

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=300):
    best_model_wts = model.state_dict()
    best_iou = 0.0
    iou_metric = JaccardIndex(task='binary', num_classes=2).to(device)
    thres = 0.25

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            # print(images.shape)
            images, masks = images.to(device), masks.to(device)
            masks = masks.bool()
            masks = masks.float()
            optimizer.zero_grad()
            outputs = model(images)
            # print("masks:",masks.shape)
            # print(torch.max(outputs))
            # print(torch.max(masks))
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        i = 0
        with torch.no_grad():
            print(len(val_loader))
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                # print(images.size())
                masks = masks.bool()
                masks = masks.float()
                # print(images.shape)
                outputs = model(images)

                # print("outputs:{}".format(outputs.shape))
                # print("masks:{}".format(masks.shape))
                ###### need to be done ########
                loss = criterion(outputs, masks)  # 计算验证集损失
                val_loss += loss.item() * images.size(0)
                iou = compute_iou(outputs, masks,thres)
                val_iou += iou.item() * images.size(0)

                # 计算测试集的loss和ioU
                print("\niou:{}".format(val_iou))
                print(torch.max(outputs))
                print(torch.min(outputs))
                bool = outputs[0]>thres
                bool = bool.float()*255
                bool = bool.permute(1,2,0)
                image = masks[0].permute(1,2,0)*255
                # print(bool.shape)
                #
                print(f"output/{i}")
                cv2.imwrite(f"output/bool_{i}.jpg", bool.cpu().numpy())
                cv2.imwrite(f"output/image_{i}.jpg", image.cpu().numpy())
                i += 1

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val IoU: {val_iou:.4f}')

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'best_model_{num_epochs}.pth')
    return model

if __name__ == '__main__':
    model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=300)

