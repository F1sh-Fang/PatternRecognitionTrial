# 人脸分类
# K-L变换
# 最近邻

from PIL import Image
import numpy as np
import json
import os

# 全局参数
folder_number = 40  # 参与分类的文件夹（人）数
image_number = 9    # 每个文件夹用于训练的人脸数

# 构造去均值样本矩阵(m, w*h), m为样本数，w、h为样本宽度和高度
def getX(path):
    X = []
    for i in range(1,folder_number+1):
        for j in range(1, image_number+1):
            path_full = path + '/ORL_Faces/s' + str(i) + '/' + str(j) + '.pgm'  # 获取图像路径
            img = np.array(Image.open(path_full))                               # 读取图像
            img = img.flatten()                                                 # 一维化
            X.append(img)                                                       # 构造图像矩阵
    mean = np.mean(X, axis=0)
    X  = np.array(X) - mean                                                     # 去均值
    return X, mean

# 通过KL变换获取特征脸
# 输入参数：去均值样本矩阵X, 重构精确度e
def getEigenfaces(X, e):
    Y = np.dot(X, np.transpose(X))                                              # 样本矩阵降维：m*wh * wh*m -> m*m
    eigenvalue, featurevector = np.linalg.eig(Y)                                # 计算特征值和特征向量(得到的是列向量构成的特征矩阵)
    featurevector = np.transpose(featurevector)                                 # 将特征向量转置一下
    eigen = []
    for i in range(0, eigenvalue.shape[0]):                                     # 将特征值和特征向量绑到一起，方便排序
        eigen.append([eigenvalue[i], featurevector[i]])
    eigen = sorted(eigen, key = lambda eigen: eigen[0], reverse=True)           # 根据特征值大小，从大到小排序
    eigenfaces_num = 0                                                          # 根据重构精度确定特征脸数目
    eigenvalue_sum = 0
    for i in range(0, eigenvalue.shape[0]):
        eigenvalue_sum += eigen[i][0]
        if eigenvalue_sum / eigenvalue.sum() >= e:
            eigenfaces_num = i + 1
            break
    eigenfaces = []
    for i in range(0, eigenfaces_num):                                          # 计算本征脸
        Y = 1 / (eigen[i][0] ** 0.5) * np.dot(eigen[i][1], X)
        eigenfaces.append(Y)
    return np.array(eigenfaces)

# 将特征脸矩阵重构为二维图像并显示
def drawEigenfaces(eigenfaces):
    os.makedirs('./image/eigenfaces', exist_ok=True)
    for name in os.listdir('./image/eigenfaces'):
        os.remove('./image/eigenfaces/' + name)
    for row in range(0, eigenfaces.shape[0]):
        # 获取本征脸
        eigenface = eigenfaces[row]
        # 调整至0-255
        # print(eigenface.max(), eigenface.min())
        eigenface = np.array((eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255, dtype='uint8')
        # reshape为二维图像
        eigenface = np.reshape(eigenface, (112, 92))
        # 保存
        img = Image.fromarray(eigenface)
        img.save('./image/eigenfaces/'+str(row+1)+'.bmp')

# 计算输入图像在特征空间的线性表示
def getCoordinate(eigenfaces, mean, img):
    img = img.flatten()                     # 一维化
    img = img - mean                        # 去均值
    coordinate = np.dot(eigenfaces, img)    # 计算线性表示
    return coordinate

# 计算所有训练图像在新特征空间的线性表示并保存至json文件
def getCoordinates(eigenfaces, mean, path):
    file_python = {'mean': mean.tolist(), 'face': [], 'eigenfaces': eigenfaces.tolist()}
    for i in range(1,folder_number+1):
        for j in range(1, image_number+1):
            path_full = path + '/ORL_Faces/s' + str(i) + '/' + str(j) + '.pgm'  # 获取图像路径
            img = np.array(Image.open(path_full))                               # 读取图像
            coordinate = getCoordinate(eigenfaces, mean, img)                   # 计算输入图像在特征空间的线性表示
            face = {'id': 's'+str(i)+'-'+str(j), 'class': 's'+str(i), 'coordinate': coordinate.tolist()}
            file_python['face'].append(face)
    # 保存为json文件wo
    os.makedirs('./image/eigenfaces', exist_ok=True)
    json_path = './image/eigenfaces/eigencoordinate.json'                       # 生成json存储路径
    # 存储为json文件
    with open(json_path, 'w') as file_json:
        json.dump(file_python, file_json, indent=2)                             # 自动换行，缩进2个空格
        print('保存完毕！')

# 读取图像和json文件，判断图像中的人脸类别
def faceClassfication(folder_path, json_path):
    # 读取json文件
    with open(json_path,'r') as file_json:
        file_python = json.load(file_json)
    eigenfaces = np.array(file_python['eigenfaces'])
    mean = file_python['mean']
    face = file_python['face']
    # 读取图像
    count = 0                                                                   # 统计分类正确次数
    for i in range(1, 41):
        img_path = folder_path + '/s' + str(i) + '.bmp'
        img = np.array(Image.open(img_path))                                    # 读取图像
        coordinate = getCoordinate(eigenfaces, mean, img)                       # 计算输入图像在特征空间的线性表示
        # 计算距离
        distences = []
        for item in face:
            face_coordinate = np.array(item['coordinate'])
            dist = np.sqrt(np.sum(np.square(face_coordinate-coordinate)))
            distences.append(dist)
        # 找到最小的距离
        index1 = distences.index(min(distences))
        face_class1 = face[index1]['class']
        if face_class1 == ('s' + str(i)):
            count += 1
        print('预测类别: {}, 实际类别: {}'.format(face_class1, 's' + str(i)))
    return count / 40

# ---------------------------------主函数---------------------------------
if __name__ == "__main__":
    # # 生成测试集
    # os.makedirs('./image/test', exist_ok=True)
    # for i in range(1,folder_number+1):
    #     path_full = './image/ORL_Faces/s' + str(i) + '/10.pgm'  # 获取图像路径
    #     img = Image.open(path_full)                             # 读取图像
    #     save_path = './image/test/s' + str(i) + '.bmp'
    #     img.save(save_path)

    # 生成特征量、特征矩阵、json文件
    path = './image'
    X, mean = getX(path)
    # print(X.shape)
    eigenfaces = getEigenfaces(X, 0.7)  # 大于0.7就准确率就不变了
    # print(eigenfaces.shape)
    drawEigenfaces(eigenfaces)
    getCoordinates(eigenfaces, mean, path)
    
    # 人脸分类测试
    img_path = './image/test'
    json_path = './image/eigenfaces/eigencoordinate.json'
    result = faceClassfication(img_path, json_path)
    print('准确率：', result)
