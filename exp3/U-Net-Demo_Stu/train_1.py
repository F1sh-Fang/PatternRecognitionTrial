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
import torchmetrics
import torch.nn.functional as F
import json

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
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

thres = 0.25

class UNetModel(nn.Module):
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, out_channels=1):
        super(UNetModel, self).__init__()
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        # 编码器部分 (Encoder)
        self.encoder1 = self.conv_block(IMG_CHANNELS, 8)
        self.encoder2 = self.conv_block(8, 16)
        self.encoder3 = self.conv_block(16, 32)
        self.encoder4 = self.conv_block(32, 64)
        self.encoder5 = self.conv_block(64, 128)

        # 解码器部分 (Decoder)
        self.decoder4_upconv = self.upconv_block(128, 64)
        self.decoder3_upconv  = self.upconv_block(64, 32)
        self.decoder2_upconv  = self.upconv_block(32, 16)
        self.decoder1_upconv  = self.upconv_block(16, 8)

        self.decoder4 = self.conv3_block(128, 64)
        self.decoder3 = self.conv3_block(64, 32)
        self.decoder2 = self.conv3_block(32, 16)
        self.decoder1 = self.conv3_block(16, 8)

        # 最终输出
        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """ 卷积块：两个卷积层 + ReLU + 最大池化 """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
        )
        return block
    def conv3_block(self, in_channels, out_channels):
        """ 卷积块：两个卷积层 + ReLU + 最大池化 """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        """ 上采样卷积块：反卷积 + 卷积块 """
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # 编码器路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        e5 = self.encoder5(F.max_pool2d(e4, 2))

        # 解码器路径
        du4 = self.decoder4_upconv(e5)
        d4 = torch.cat((du4, e4), dim=1)
        dd4 = self.decoder4(d4)
        du3 = self.decoder3_upconv(dd4)
        d3 = torch.cat((du3, e3), dim=1)
        dd3 = self.decoder3(d3)
        du2 = self.decoder2_upconv(dd3)
        d2 = torch.cat((du2, e2), dim=1)
        dd2 = self.decoder2(d2)
        du1 = self.decoder1_upconv(dd2)
        d1 = torch.cat((du1, e1), dim=1)
        dd1 = self.decoder1(d1)

        # 最终输出
        out = self.final_conv(dd1)
        out = torch.sigmoid(out)
        return out


model = UNetModel(128, 128, 3).to(device)
criterion = nn.BCELoss().to(device) # nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=300):
    best_model_wts = model.state_dict()
    best_iou = 0.0
    loss_list = []

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
            # print(len(val_loader))
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

                thres = (torch.max(outputs) - torch.min(outputs))*1/2 + torch.min(outputs)
                bool = outputs[0] > thres
                bool = bool.float()
                iou = torchmetrics.functional.classification.binary_jaccard_index(outputs, masks, thres.item())# iou_metric(outputs, masks, thres)# compute_iou(outputs, masks)
                val_iou += iou.item() * images.size(0)

                # 计算测试集的loss和ioU
                # print("\niou:{}".format(val_iou))
                # print(torch.max(outputs))
                # print(torch.min(outputs))
                bool = bool.float()
                bool = bool * 255
                bool = bool.permute(1,2,0)
                image = masks[0].permute(1,2,0)*255
                # print(bool.shape)
                #
                # print(f"output/{i}")
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

        loss_list.append([epoch, train_loss, val_loss, val_iou])

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            best_model_wts = model.state_dict()

        if epoch % 20 == 0 and epoch != 0:
            torch.save(model.state_dict(), f'models/model_{epoch}.pth')
            with open(f'models/model_loss_{epoch}.json', 'w') as file:
                json.dump(loss_list, file)

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'best_model_{num_epochs}.pth')

    with open('loss.json','w') as file:
        json.dump(loss_list, file)

    loss_array = np.array(loss_list)
    x_epoch = np.array(loss_array[:, 0])
    y_trainloss = np.array(loss_array[:, 1])
    y_valloss = np.array(loss_array[:, 2])
    y_valiou = np.array(loss_array[:, 3])
    plt.plot(x_epoch,y_trainloss)
    plt.plot(x_epoch, y_valloss)
    plt.plot(x_epoch, y_valiou)
    plt.xlabel('epoch')
    plt.ylabel('loss/iou')
    plt.grid()
    plt.legend(['train_loss', 'val_loss', 'val_iou'])
    plt.show()

    return model



if __name__ == '__main__':

    model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=300)

