import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CharacterDataset
from PIL import Image
import os
import shutil

# 基本参数设置
epochs = 100
batch_size = 256
number_classes = 34  # 字符类别数（34类字符）
epoch_best = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查设备是否支持GPU


# K-L变换（特征提取）
def KL_transform(images, mean_image, eigenvectors):
    """
    对图像进行K-L变换
    images: 输入图像张量，形状为(batch_size, 28, 28)
    mean_image: 图像数据集的均值图像
    eigenvectors: 特征向量矩阵
    """
    images = images.view(images.size(0), -1)  # Flatten images to (batch_size, 784)
    normalized_images = images - mean_image  # 减去均值图像
    eigenvectors = eigenvectors.to(torch.float32)
    features = torch.matmul(normalized_images, eigenvectors)  # 计算K-L变换特征
    return features


# 计算图像数据集的均值图像和特征向量
def compute_KL_features(dataset, image_size=28, e=0.95):
    images = []
    for image_path, _ in dataset.image_paths:
        image = Image.open(image_path).convert('L')
        image = transforms.ToTensor()(image).view(-1)  # Flatten image
        images.append(image.numpy())

    images = np.array(images,dtype=np.float32)
    mean_image = np.mean(images, axis=0)  # 计算均值图像
    centered_images = images - mean_image  # 对图像进行中心化处理
    cov_matrix = np.cov(centered_images.T)  # 计算协方差矩阵
    eigvals, eigvecs = np.linalg.eig(cov_matrix)  # 求解特征值和特征向量
    eigen = []
    for i in range(0, eigvals.shape[0]):  # 将特征值和特征向量绑到一起，方便排序
        eigen.append([eigvals[i], eigvecs[i]])
    eigen = sorted(eigen, key=lambda eigen: eigen[0], reverse=True)  # 根据特征值大小，从大到小排序
    eigen_num = 0  # 根据重构精度确定特征脸数目
    eigenvalue_sum = 0
    for i in range(0, eigvals.shape[0]):
        eigenvalue_sum += eigen[i][0]
        if eigenvalue_sum / eigvals.sum() >= e:
            eigen_num = i + 1
            break
    eigenvecs = []
    for i in range(0, eigen_num):  # 计算本征脸
        eigenvecs.append(eigen[i][1])
    eigenvecs = np.array(eigenvecs)
    eigenvecs = eigenvecs.T  # 转置特征向量矩阵
    return mean_image, eigenvecs


# ------------------------------------------------------------ 定义网络结构 ------------------------------------------------------------
class Classification(nn.Module):
    def __init__(self, input_size):
        super(Classification, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(input_size, 100),  # 输入层到隐藏层（100个神经元）
            nn.ReLU(True),
            nn.Linear(100, number_classes),  # 隐藏层到输出层（34类输出）
        )

    def forward(self, x):
        output = self.output(x)
        return output


# ------------------------------------------------------------ 数据加载与处理 ------------------------------------------------------------
# 载入数据
print('载入数据……')
path = './dataset/train.txt'
transform = transforms.Compose([transforms.ToTensor(), ])  # 仅将图片转化为Tensor
characters_train = CharacterDataset(path, transform=transform)
data_loader_train = DataLoader(characters_train, batch_size=batch_size, shuffle=True)

# 计算K-L特征
mean_image, eigenvectors = compute_KL_features(characters_train)

# ------------------------------------------------------------ 网络训练 ------------------------------------------------------------
print('构建模型……')
input_size = eigenvectors.shape[1]  # 输入大小由K-L特征维度决定
CC = Classification(input_size).to(device)
lossFun = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adadelta(CC.parameters())

# # 模型训练
# loss_epochs_train = []  # 训练过程中每个epoch的平均损失
# loss_epochs_test = []
# epochs_x = []  # 显示用的横坐标
# print('开始训练……')
# for epoch in range(epochs):
#     CC.train()  # 设置模型为训练模式
#     loss_train = []
#     for index, (images, labels) in enumerate(data_loader_train):
#         images, labels = images.to(device), labels.to(device)  # 数据迁移到GPU
#         # K-L变换特征提取
#         features = KL_transform(images, torch.tensor(mean_image).to(device), torch.tensor(eigenvectors).to(device))
#         CC_output = CC(features)  # 前向传播
#         loss = lossFun(CC_output, labels)  # 计算损失
#         optimizer.zero_grad()  # 梯度清零
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#         loss_train.append(loss.item())
#         print(f'epoch: {epoch + 1}, batch: {index + 1}, loss: {loss.item():.6f}')
#     # 保存模型权重
#     if (epoch + 1) % 10 == 0:
#         torch.save(CC.state_dict(), f'./models/CC_{epoch + 1}.pth')
#         torch.save(optimizer.state_dict(), f'./models/CC_optimizer_{epoch + 1}.pth')
#
#     # 记录损失
#     loss_epochs_train.append(np.mean(loss_train))
#     epochs_x.append(epoch + 1)
#
# # 最终模型保存
# torch.save(CC.state_dict(), './models/CC_final.pth')
# torch.save(optimizer.state_dict(), './models/CC_optimizer_final.pth')

# ------------------------------------------------------------ 网络测试 ------------------------------------------------------------
# 模型实例化
CC.eval()  # 设置模型为评估模式
weights_net = torch.load('./models/CC_95.pth', map_location=device)
CC.load_state_dict(weights_net)

softmax = nn.Softmax(dim=1)  # 用于概率计算
# 构建结果索引
index_dict = {i: chr(48 + i) if i < 10 else chr(65 + i - 10) for i in range(34)}  # 生成字符标签字典

# 测试集分类测试
os.makedirs('./output', exist_ok=True)
for name in os.listdir('./output'):
    path = './output/' + name
    os.remove(path)
test_path = './dataset/test.txt'
number = 0  # 测试样本总数
error_num = 0  # 错误分类的样本数
rate = 0  # 分类准确率
with open(test_path, 'r', encoding='utf-8') as test:
    for line in test.readlines():
        line = line.rstrip()
        words = line.split()
        label = words[-1]
        path = line[:-(len(label) + 1)]
        label = int(label)
        image_name = path.split('/')[-1]
        image = Image.open(path).convert('L')
        image = transform(image).view(1, -1).to(device)  # 图像转为1D Tensor
        features = KL_transform(image, torch.tensor(mean_image).to(device), torch.tensor(eigenvectors).to(device))
        output = softmax(CC(features))  # 前向传播
        index = output.topk(1)[1].cpu().numpy()[0][0]  # 获取预测结果
        if label != index:
            error_num += 1
            print(f'Wrong! label: {index_dict[label]}, prediction: {index_dict[index]}, image_name: {image_name}')
            target_path = f'./output/{index_dict[label]}_{index_dict[index]}_{image_name}'
            shutil.copyfile(path, target_path)  # 错误分类样本保存
        number += 1
print(f'测试图像数量: {number}, 误分类数量: {error_num}, 分类准确率: {(1 - error_num / number) * 100:.2f}%')
