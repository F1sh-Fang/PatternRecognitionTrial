# 人脸分类
# 基于类内类间离散度的特征降维方法
# 最近邻

from PIL import Image
import numpy as np
import json
import os

# 全局参数
folder_number = 40  # 参与分类的文件夹（人）数
image_number = 9    # 每个文件夹用于训练的人脸数

# 构造样本矩阵(m, w*h), m为样本数，w、h为样本宽度和高度
def getX(path):
    X = []
    means = []
    for i in range(1,folder_number+1):
        X_temp = []
        for j in range(1, image_number+1):
            path_full = path + '/ORL_Faces/s' + str(i) + '/' + str(j) + '.pgm'  # 获取图像路径
            img = np.array(Image.open(path_full))                               # 读取图像
            img = img.flatten()                                                 # 一维化
            X_temp.append(img)                                                  # 构造图像矩阵
        X.append(X_temp)
        means.append(np.mean(X_temp, axis=0))                                    # 计算类内均值向量
    return np.array(X), np.array(means)

# 计算类内离散度矩阵
def getSw(X, means):
    Sw = np.zeros((X.shape[2], X.shape[2]))
    for index in range(X.shape[0]):
        samples = X[index]
        mean = means[index]
        for vec in samples:
            vec = (vec-mean).reshape((-1, 1))
            # vec_T = vec.reshape((1, -1))
            Sw += np.dot(vec, vec.T)
    num = X.shape[0] * X.shape[1]   # 样本数
    Sw /= num
    return Sw

# 计算类间离散度矩阵
def getSb(X, means):
    num = X.shape[0] * X.shape[1]   # 样本数
    mean = 0                        # 所有样本的加权均值
    for index in range(X.shape[0]):
        mean += means[index] *  X[index].shape[0] / num
    Sb = np.zeros((X.shape[2], X.shape[2]))
    for samples in X:
        for vec in samples:
            vec = (vec-mean).reshape((-1, 1))
            # vec_T = vec.reshape((1, -1))
            Sb += np.dot(vec, vec.T) * samples.shape[0]
    Sb /= num
    return Sb

# 计算特征变换阵
def getW(Sw, Sb, e, num=None):
    invSw = np.linalg.inv(Sw)
    eigenvalue, featurevector = np.linalg.eig(np.dot(invSw, Sb))                # 计算invSwSb的特征值、特征向量
    eigen = []
    for i in range(0, eigenvalue.shape[0]):                                     # 将特征值和特征向量绑到一起，方便排序
        eigen.append([eigenvalue[i], featurevector[i]])
    eigen = sorted(eigen, key = lambda eigen: eigen[0], reverse=True)           # 根据特征值大小，从大到小排序
    eigenfaces_num = 0                                                          # 根据重构精度确定特征向量数目
    eigenvalue_sum = 0
    if num == None:
        for i in range(0, eigenvalue.shape[0]):                                     # 确定特征变换阵的维度
            eigenvalue_sum += eigen[i][0]
            if eigenvalue_sum / eigenvalue.sum() >= e:
                eigenfaces_num = i + 1
                break
    else:
        eigenfaces_num = num
    W = []
    print(eigenfaces_num)
    for i in range(0, eigenfaces_num):
        W.append(eigen[i][1])                                                   # 构造特征变换阵d*wh
    return np.real(W)                                                           # 取实部

# 计算输入图像在特征空间的线性表示
def getCoordinate(W, img):
    img = img.flatten()                                                         # 一维化
    coordinate = np.dot(W, img)                                                 # 计算线性表示
    return coordinate

# 计算所有训练图像在新特征空间的线性表示并保存至json文件
def getCoordinates(W, path):
    file_python = {'face': [], 'W': W.tolist()}
    for i in range(1,folder_number+1):
        for j in range(1, image_number+1):
            path_full = path + '/ORL_Faces/s' + str(i) + '/' + str(j) + '.pgm'  # 获取图像路径
            img = np.array(Image.open(path_full))                               # 读取图像
            coordinate = getCoordinate(W, img)                                  # 计算输入图像在特征空间的线性表示
            face = {'id': 's'+str(i)+'-'+str(j), 'class': 's'+str(i), 'coordinate': coordinate.tolist()}
            file_python['face'].append(face)
    # 保存为json文件
    os.makedirs('./image/matrix', exist_ok=True)
    json_path = './image/matrix/matrixs.json'                                   # 生成json存储路径
    # 存储为json文件
    with open(json_path, 'w') as file_json:
        json.dump(file_python, file_json, indent=2)                             # 自动换行，缩进2个空格
        print('保存完毕！')

# 读取图像和json文件，判断图像中的人脸类别
def faceClassfication(folder_path, json_path):
    # 读取json文件
    with open(json_path,'r') as file_json:
        file_python = json.load(file_json)
    W = np.array(file_python['W'])
    face = file_python['face']
    # 读取图像
    count = 0                                                                   # 统计分类正确次数
    for i in range(1, 41):
        img_path = folder_path + '/s' + str(i) + '.bmp'
        img = np.array(Image.open(img_path))                                    # 读取图像
        coordinate = getCoordinate(W, img)                                      # 计算输入图像在特征空间的线性表示
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

if __name__ == "__main__":
    # # 生成测试集
    # os.makedirs('./image/test', exist_ok=True)
    # for i in range(1,folder_number+1):
    #     path_full = './image/ORL_Faces/s' + str(i) + '/10.pgm'  # 获取图像路径
    #     img = Image.open(path_full)                             # 读取图像
    #     save_path = './image/test/s' + str(i) + '.bmp'
    #     img.save(save_path)

    # # 生成特征量、特征矩阵、json文件
    # path = './image'
    # X, means = getX(path)
    # Sw = getSw(X, means)        # 100秒
    # Sb = getSb(X, means)        # 123秒
    # W = getW(Sw, Sb, 0.8, num=20)
    # getCoordinates(W, path)

    # 人脸分类测试
    img_path = './image/test'
    json_path = './image/matrix/matrixs.json'
    result = faceClassfication(img_path, json_path)
    print('准确率：', result)
