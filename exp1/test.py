# import cv2
import numpy as np
import os

# m = np.array([[1, 2], [3, 4], [5, 6]], np.uint8)
# print(m.shape)

# # numpy数组变为一维向量
# n = m.flatten()
# # n = np.reshape(m, (1,6))
# print(n.shape)
# print(n)

# # numpy数组添加元素
# m = np.array([], np.uint8)
# item = [[7, 8]]
# m = np.append(m, item)
# m = m - 1
# print(type(item))
# print(m.sum())
# # print(m.shape[0])
# # print(m)

# # 计算矩阵特征值特征向量
# # X = np.array([[-1,1,0], [-4,3,0], [1,0,2]])
# Y = np.array([[3, 2, 4], [2, 0, 2], [4, 2, 3]])
# # # print("X=",X)
# # # print(X.shape)

# eigenvalue, featurevector=np.linalg.eig(Y)
# print(eigenvalue)
# for item in featurevector:
#     print(item)
# # print("eigenvalue=",eigenvalue.shape)
# print("featurevector=\n",featurevector)
# # print(type(eigenvalue))
# # print(type(featurevector))

# # 矩阵乘法
# X = [1, 2, 3]
# Y = np.array([[5, 6], [7, 8]])
# Y = np.append(Y, [[9, 10]], axis=0)
# print(X)
# print(Y)

# print(np.dot(X, Y))


# # numpy元素排序
# X = [[2, [1, 2, 3]], [1, [2, 3, 4]], [4, [3, 4, 5]]]
# # X = np.array(X)
# # print(X.shape)
# # X = np.append(X, [[3, [1, 3, 4]]], axis=0)
# X = sorted(X, key = lambda X: X[0], reverse=True)
# print(X)

# X = np.array([1, 2, 3])
# print(X.mean())

# print(4 ** 0.5)
# for i in range(0, 10):
#     img = cv2.imread('./image/eigenfaces/'+str(i+1)+'.bmp')
#     cv2.imshow('eigenface', img)
#     cv2.waitKey(0)

# # 创建文件夹
# os.makedirs('./test/one', exist_ok=True)  # 若存在则不创建


# # ----------------------------------------------------------------------------------------降维特征空间坐标保存和读取json文件------------------------------------------------------------------------
# import json

# # 生成python字典
# file_python = [{'id':'s101', 'class':'s1', 'coordinate':[1, 2, 3]}, {'id':'s102', 'class':'s1', 'coordinate':[4, 5, 6]}]
# # 生成json存储路径
# json_path = './image/eigenfaces/eigencoordinate.json'
# # 存储为json文件
# with open(json_path, 'w') as file_json:
#     json.dump(file_python, file_json, indent=2) # 自动换行，缩进2个空格
#     print('保存完毕！')


# # 生成标签访问路径
# json_path = './image/eigenfaces/eigencoordinate.json'
# # 载入json文件
# with open(json_path,'r') as file_json:
#     file_python = json.load(file_json)
#     print(file_python)


# # -------------------------------------------------------------------------------------------------------计算向量距离------------------------------------------------------------------------------------
# import numpy as np
# x=np.array([1,2])
# # x = [1, 2]
# y=np.array([[3,4], [1, 2]])
# # print(y)
# print(np.sum(y, axis=0) / 2)

# print(type(y - x))

# #方法一：根据公式求解
# d1=np.sqrt(np.sum(np.square(x-y)))
# print(d1)
# #方法二：根据scipy库求解
# from scipy.spatial.distance import pdist
# X=np.vstack([x,y])
# d2=pdist(X)
# print(d2)

# # 矩阵计算
# a = np.array([1, 2, 3, 4, 5, 6]).reshape((2,3))
# b = np.array([1, 2, 3]).reshape((3, 1))

# c = np.dot(b, a)
# print(c)

# # a = np.eye(5)
# # a = np.matrix(a)
# # print(a, a.I)
# # b = np.linalg.inv(a)
# # print(b)

# import json
# W = [[1-2j, 2+1j, 3], [4, 5, 6], [7, 8, 9]]
# W = np.real(W)
# coordinate = np.array([1, 2, 3])
# print(W)
# file_python = {'matrixW': W.tolist(), 'face': []}
# face = {'id': 's1-1', 'class': 's1', 'coordinate': coordinate.tolist()}
# file_python['face'].append(face)
# # 保存为json文件
# os.makedirs('./image/matrix', exist_ok=True)
# json_path = './image/matrix/matrixs.json'                                   # 生成json存储路径
# # 存储为json文件
# with open(json_path, 'w') as file_json:
#     json.dump(file_python, file_json, indent=2)                             # 自动换行，缩进2个空格
#     print('保存完毕！')

a = np.array([1, 2, 3, 4]).reshape((2, 2))
b = np.mean(a, axis=0)
print(a, b)