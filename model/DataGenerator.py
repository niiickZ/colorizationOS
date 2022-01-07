import os
import numpy as np
import random

class DataGenerator:
    def adjustSize(self, dataSet, batch_size):
        """ 令数据集大小可以被batch size整除"""
        fnum = len(dataSet)
        fnum = fnum - fnum % batch_size
        dataSet = dataSet[:fnum]
        random.shuffle(dataSet)

        return dataSet

    def convertToOnehot(self, labelList, num_class):
        """ 将整数标转换为one-hot编码"""
        labels = np.eye(num_class)[labelList]  # 转换为one-hot编码
        labels = labels.astype(np.int_)

        # 转换为np.ndarray型的list并返回
        return list(labels)

    @classmethod
    def flowFromDirectory(cls, dataDir, batch_size):
        """ 获取目录下每张图片绝对路径及对应one-hot编码
            目录结构：
            dataDir
                |- class1
                |   |- img1
                |   |- img2
                |   |- ...
                |- class2
                |   |- img1
                |   |- img2
                |   |- ...
                |- ...
        """
        fileList = []  # 每张图片绝对路径
        labelList = []  # 每张图片对应one-hot编码

        num_class = 0

        num = 0
        for root, dirs, files in os.walk(dataDir):
            if os.path.samefile(root, dataDir):  # 根目录
                num_class = len(dirs)
                continue

            fpath = os.path.abspath(root)
            for fname in files:
                fileList.append(os.path.join(fpath, fname))  # 将完整路径加入文件路径列表

            labelList.extend([num] * len(files))  # 字符串标签(即目录名)转换为整数标签
            num += 1

        labels = cls().convertToOnehot(labelList, num_class)  # 整数标签转换为one-hot
        dataSet = list(zip(fileList, labels))  # 将路径与对应one-hot标签压缩为元组
        dataSet = cls().adjustSize(dataSet, batch_size)

        return dataSet, num_class

    @classmethod
    def flowFromFile(cls, fpath, root, batch_size):
        """ 根据文件中图片训练集路径列表，获取每张图片绝对路径及对应one-hot编码
            文件保存结构：
            root
                |- train
                |   |- class1
                |   |   |- img1
                |   |   |- img2
                |   |   |- ...
                |   |- class2
                |   |   |- img1
                |   |   |- img2
                |   |   |- ...
                |- val
                |   |- ...

            文件列表保存结构:
            train/class1/img1
            train/class1/img2
            train/class1/...
            train/class2/img1
            train/class2/img2
            train/class2/...
            ...
        """
        fileList = []  # 每张图片绝对路径
        labelList = []  # 每张图片对应one-hot编码

        # 获取train下所有目录名(即标签名)
        dirList = os.listdir(os.path.join(root, 'train'))
        num_class = len(dirList)

        # 将字符串标签(即目录名)映射为整数标签
        labelMap = {}
        for i in range(len(dirList)):
            labelMap[dirList[i]] = i

        with open(fpath, 'r') as dataListFile:
            for img_path in dataListFile:
                img_path = img_path.replace('\n', '').replace('\r', '')
                dirName = os.path.basename(os.path.dirname(img_path))  # 'train/dirname/img' -> 'dirname'

                fileList.append(os.path.join(root, img_path))  # 将完整路径加入文件路径列表
                labelList.append(labelMap[dirName])  # 根据映射加入整数类别标签

        labels = cls().convertToOnehot(labelList, num_class)  # 整数标签转换为one-hot
        dataSet = list(zip(fileList, labels))  # 将路径与对应one-hot标签压缩为元组
        dataSet = cls().adjustSize(dataSet, batch_size)

        return dataSet, num_class