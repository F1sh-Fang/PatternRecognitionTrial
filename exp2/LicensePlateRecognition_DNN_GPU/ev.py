import numpy as np
import torch
# 模型构建
import torch.nn as nn
# 模型训练
import torch.optim as optim
# 数据准备
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CharacterDataset

# --- 基本参数设置 ---
epochs = 50
batch_size = 256
number_classes = 34                                         # 分割类别（含背景类）
epoch_best = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查设备是否支持GPU

# ------------------------------------------------------------ 定义网络结构 ------------------------------------------------------------
print('构建模型……')
class Classification(nn.Module):
    # 定义构造函数
    def __init__(self):
        super(Classification, self).__init__()
        # 定义网络结构
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, number_classes),
        )
    # 定义前向传播函数
    def forward(self, x):
        output = self.output(x)
        return output

# ------------------------------------------------------------ 网络训练 ------------------------------------------------------------
CC = Classification().to(device)                                   # 模型迁移到GPU
lossFun = nn.CrossEntropyLoss().to(device)                         # 损失函数迁移到GPU
optimizer = optim.Adadelta(CC.parameters())                        # 定义优化器

# # 载入并更新网络、优化器权重参数
# weights_net = torch.load('./models/CC_final.pth', map_location=device)
# CC.load_state_dict(weights_net)
# weights_optimizer = torch.load('./models/CC_optimizer_final.pth', map_location=device)
# optimizer.load_state_dict(weights_optimizer)

# 训练数据准备
# print('载入数据……')
# path = './dataset/train.txt'
# transform = transforms.Compose([transforms.ToTensor(), ])
# characters_train = CharacterDataset(path, transform=transform)
# data_loader_train = DataLoader(characters_train, batch_size=batch_size, shuffle=True)
#
# # 测试数据准备
# path = './dataset/test.txt'
# transform = transforms.Compose([transforms.ToTensor(), ])
# characters_test = CharacterDataset(path, transform=transform)
# data_loader_test = DataLoader(characters_test, batch_size=batch_size, shuffle=False)
#
# # 模型训练
# loss_epochs_train = []                                  # 训练过程中每个epoch的平均损失保存在此列表中，用于显示
# loss_epochs_test = []
# epochs_x = []                                           # 显示用的横坐标
# print('开始训练……')
# for epoch in range(epochs):
#     CC.train()  # 设置模型为训练模式
#     loss_train = []
#     for index, (images, labels) in enumerate(data_loader_train):
#         images, labels = images.to(device), labels.to(device)       # 数据迁移到GPU
#         CC_output = CC(images)                                      # 前向传播
#         loss = lossFun(CC_output, labels)                           # 计算损失
#         optimizer.zero_grad()                                       # 梯度清零
#         loss.backward()                                             # 反向传播
#         optimizer.step()                                            # 更新参数
#         loss_train.append(loss.data.item())
#         print('epoch:{}, batch:{}, loss:{:.6f}'.format(epoch+1, index+1, loss.data.item()))
#     if (epoch+1) % 5 == 0:  # 每训练10个epoch保存一次模型
#         torch.save(CC.state_dict(), './models/CC_'+str(epoch+1)+'.pth')
#         torch.save(optimizer.state_dict(), './models/CC_optimizer_'+str(epoch+1)+'.pth')
#
#     # 模型测试（每个epoch使用测试集进行一次验证）
#     CC.eval()  # 设置模型为评估模式
#     loss_test = []
#     print('开始验证……')
#     for index, (images, labels) in enumerate(data_loader_test):
#         images, labels = images.to(device), labels.to(device)       # 数据迁移到GPU
#         CC_output = CC(images)                                      # 前向传播
#         loss = lossFun(CC_output, labels)                           # 计算损失
#         loss_test.append(loss.data.item())
#     print('平均损失：', np.mean(loss_test))
#
#     # 记录损失
#     loss_epochs_train.append(np.mean(loss_train))
#     loss_epochs_test.append(np.mean(loss_test))
#     epochs_x.append(epoch+1)
#
# # 最终模型保存
# torch.save(CC.state_dict(), './models/CC_final.pth')
# torch.save(optimizer.state_dict(), './models/CC_optimizer_final.pth')
#
# # # 损失可视化（需要安装matplotlib库，安装指令：conda install matplotlib）
# # import matplotlib.pyplot as plt
# # plt.figure('loss')
# # plt.plot(epochs_x, loss_epochs_train, label='Training loss')
# # plt.plot(epochs_x, loss_epochs_test, label='Validation loss')
# # plt.xlabel('epochs')
# # plt.ylabel('loss')
# # plt.legend(frameon=False)
# # plt.show()
#
# epoch_best = int(loss_epochs_test.index(min(loss_epochs_test)) / 10 + 0.6) * 10
# if epoch_best < 10:
#     epoch_best = 10
# print('epoch_best:', epoch_best)

# ------------------------------------------------------------ 网络测试 ------------------------------------------------------------
# 识别测试
import os
import shutil
from PIL import Image

# 模型实例化
CC = Classification().to(device)  # 模型迁移到GPU
CC.eval()  # 设置模型为评估模式
# 载入并更新网络权重参数
weights_net = torch.load('./models/CC_20.pth', map_location=device)  # 从文件加载权重
CC.load_state_dict(weights_net)  # 将权重加载到模型中

softmax = nn.Sequential(nn.Softmax(dim=1))  # 定义 softmax 用于概率计算
# 构建结果索引
index_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'J', 19:'K',
    20:'L', 21:'M', 22:'N', 23:'P', 24:'Q', 25:'R', 26:'S', 27:'T', 28:'U', 29:'V',
    30:'W', 31:'X', 32:'Y', 33:'Z'
}
# 测试集分类测试
os.makedirs('./output', exist_ok=True)  # 创建输出目录
for name in os.listdir('./output'):
    path = './output/' + name
    os.remove(path)  # 清空输出目录
test_path = './dataset/test.txt'
number = 0  # 测试样本总数
error_num = 0  # 分类错误样本数
rate = 0  # 分类准确率
with open(test_path, 'r', encoding='utf-8') as test:
    for line in test.readlines():
        line = line.rstrip()
        words = line.split()
        label = words[-1]
        path = line[:-(label.__len__()+1)]
        label = int(label)
        image_name = path.split('/')[-1]

        image = Image.open(path).convert('L')  # 加载并转换图像为灰度
        transform = transforms.Compose([transforms.ToTensor(), ])  # 定义数据转换
        image = transform(image).view(1, 1, 28, 28).to(device)  # 数据迁移到GPU
        output = softmax(CC(image))  # 前向传播
        index = output.topk(1)[1].cpu().numpy()[0][0]  # 获取预测结果并迁移回CPU
        if label != index:
            error_num += 1
            print('Wrong! label: {}, prediction: {}, image_name: {}'.format(index_dict[label], index_dict[index], image_name))
            target_path = './output/' + index_dict[label] + '_' + index_dict[index] + '_' +image_name
            shutil.copyfile(path, target_path)  # 将错误分类的样本复制到输出目录
        number += 1
print('测试图像数量: {}, 误分类数量: {}, 分类准确率: {:.2f}%'.format(number, error_num, (1-error_num/number)*100))
