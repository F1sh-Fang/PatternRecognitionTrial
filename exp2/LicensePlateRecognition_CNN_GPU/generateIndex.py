# 生成训练集和测试集索引文件
import os
import numpy as np

root = '../license_plate_number_dataset'
train_path = root + '/Chars74K_train'
test_path  = root + '/Chars74K_test'

# 获取训练集文件列表
train_folder_names = os.listdir(train_path)

# 获取测试集文件列表
test_folder_names  = os.listdir(test_path)

# 存储训练集
with open(root+'/train.txt', 'w', encoding='utf-8') as train:
    for folder_name in train_folder_names:
        folder_path = train_path + '/' + folder_name
        image_names = os.listdir(folder_path)
        for image_name in image_names:
            image_path = folder_path + '/' + image_name
            label = folder_name
            train.write(image_path + ' ' + label + '\n')

# 存储测试集
with open(root+'/test.txt', 'w', encoding='utf-8') as test:
    for folder_name in test_folder_names:
        folder_path = test_path + '/' + folder_name
        image_names = os.listdir(folder_path)
        for image_name in image_names:
            image_path = folder_path + '/' + image_name
            label = folder_name
            test.write(image_path + ' ' + label + '\n')
