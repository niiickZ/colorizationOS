import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, \
    Input, BatchNormalization, Activation, Layer
from tensorflow.keras.callbacks import Callback
import numpy as np
import random
import cv2
from .DataGenerator import DataGenerator

class ImageProcessor:
    @staticmethod
    def norm_L(x):
        """ 归一化。 L空间值域 L:[0,100], opencv中处理为 L*2.55"""
        return (x / 2.55 - 50.) / 50.

    @staticmethod
    def norm_ab(x):
        """ 归一化。 a(b)空间值域:[-127,128], opencv中处理为, a(b)+127"""
        return (x - 127.) / 128.

    @staticmethod
    def unNorm_L(x):
        """ 去归一化 L空间"""
        return (x * 50. + 50.) * 2.55

    @staticmethod
    def unNorm_ab(x):
        """ 去归一化 a(b)空间"""
        return x * 128. + 127.

class MergeLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, localFeat, globalFeat):
        shape = tf.shape(localFeat)[:]
        batch_size, w, h = shape[0], shape[1], shape[2]
        globalFeat = tf.reshape(globalFeat, (batch_size, 1, 1, 256))  # (None, 256) -> (None, 1, 1, 256)
        globalFeatTile = tf.tile(globalFeat, [1, w, h, 1])  # 重复全局特征

        jointFeat = tf.concat([localFeat, globalFeatTile], axis=-1)  # 融合

        return jointFeat

class ColorizationNet:
    def __init__(self):
        self.output_classes = 365

        self.low_levelNet = self.buildLowNet()
        self.mid_levelNet = self.buildMidNet()
        self.globalFeatNet, self.classifyNet = self.buildGlobalNet()
        self.colorizationNet = self.buildColorizeNet()
        self.model = self.buildNet()

    def buildNet(self):
        """ 构建整体网络
            input: img_org -> low-level net -> mid-level net -> output: local features
            input: img_scale -> low-level net -> global features net -> output: global features & class label

            fusion: local features & global features -> colorization net -> output: ab spaces
        """
        img_org = Input(shape=(None, None, 1))
        localFeat = self.mid_levelNet(self.low_levelNet(img_org))  # 局部特征

        img_scale = Input(shape=(224, 224, 1))
        low_levelFeat = self.low_levelNet(img_scale)
        globalFeat = self.globalFeatNet(low_levelFeat)  # 全局特征
        category = self.classifyNet(low_levelFeat)  # 分类结果

        # 融合层
        jointFeat = MergeLayer()(localFeat, globalFeat)

        output_img = self.colorizationNet(jointFeat)  # 色度上色结果

        model = Model([img_org, img_scale], [output_img, category])
        return model

    def buildLowNet(self):
        """ low-level features network
            C64s2 - C128 - C128s2 - C256 - C256s2 - C512
        """
        input_img = Input(shape=(None, None, 1))

        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        output_feat = Activation('relu')(x)

        return Model(input_img, output_feat, name='lowFeaturesNet')

    def buildMidNet(self):
        """ mid-level features network (compute local features)
            C512 - C256
        """
        low_levelFeat = Input(shape=(None, None, 512))

        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(low_levelFeat)
        localFeat = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)

        return Model(low_levelFeat, localFeat, name='midFeaturesNet')

    def buildGlobalNet(self):
        """ global features network and classification network
            common: C512s2 - C512 - C512s2 - C512 - FC1024 - FC512 -
            global features: - FC256
            classification: - FC256 - FC num_classes
        """
        low_levelFeat = Input(shape=(28, 28, 512))

        x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(low_levelFeat)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        flat = Flatten()(x)
        x = Dense(1024, activation='relu')(flat)
        x = Dense(512, activation='relu')(x)

        # 分类隐藏层
        classifyLayer = Dense(256, activation='relu')(x)
        classifyResult = Dense(self.output_classes, activation='softmax')(classifyLayer)

        # 全局特征
        globalFeat = Dense(256, activation='relu')(x)

        return Model(low_levelFeat, globalFeat, name='globalFeaturesNet'), \
               Model(low_levelFeat, classifyResult, name='classification')

    def buildColorizeNet(self):
        """ colorization network
            C256 - C128 - upsample 2x - C64 - C64 - upsample 2x - C32 - C2 - upsample 2x
        """
        input_feat = Input(shape=(None, None, 512))

        x = Conv2D(filters=256, kernel_size=3, padding='same')(input_feat)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=2, kernel_size=3, padding='same', activation='tanh')(x)  # sigmoid -> tanh

        output_img = UpSampling2D()(x)

        return Model(input_feat, output_img, name='colorization')

    def colorize(self, img_gray):
        """ 灰度图上色
            :param img_gray 3通道灰度图
        """
        def getInput(img):
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # 转换为Lab
            img_L = img_lab[:, :, :1].astype(np.float32)  # 提取亮度L通道
            img_L = np.expand_dims(img_L, 0)  # (h, w, 1) -> (1, h, w, 1)
            return ImageProcessor.norm_L(img_L)  # L通道归一化

        height, width, channel = img_gray.shape
        img_scale = cv2.resize(img_gray, (224, 224))

        input_L = getInput(img_gray)
        input_L_scale = getInput(img_scale)

        output_L = ImageProcessor.unNorm_L(input_L)
        output_ab = self.model.predict([input_L, input_L_scale])[0]  # 预测ab通道
        output_ab = ImageProcessor.unNorm_ab(output_ab[0])  # ab通道去归一化

        output_Lab = np.zeros((height, width, 3))
        output_Lab[:, :, :1] = output_L
        output_Lab[:, :, 1:] = output_ab
        output_Lab = output_Lab.astype(np.uint8)

        img_bgr = cv2.cvtColor(output_Lab, cv2.COLOR_Lab2BGR)  # 转换回bgr
        return img_bgr

class ModelSaver(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights('/mnt/weights{}.h5'.format(epoch))

class ColorizationModel(ColorizationNet):
    def __init__(self):
        super().__init__()

    def dataIter(self, batch_size):
        """ 数据生成器
            生成训练数据尺寸为 224x224
        """
        def processImg(img):
            """ 图片增强
                先调整图片大小为 256x256, 再应用随机裁剪与随机水平翻转, 调整尺寸至 224x224
            """
            img = cv2.resize(img, (256, 256))

            opt = random.randint(0, 1)
            img = img if opt else cv2.flip(img, 1)  # 随机水平翻转

            opt = random.randint(0, 1)
            img = img[32:, :, :] if opt else img[:224, :, :]  # 随机上/下裁剪

            opt = random.randint(0, 1)
            img = img[:, 32:, :] if opt else img[:, :224, :]  # 随机左/右裁剪

            return img

        img_row = 224
        img_col = 224

        img_L = np.zeros((batch_size, img_row, img_col, 1))
        img_ab = np.zeros((batch_size, img_row, img_col, 2))
        output_labels = np.zeros((batch_size, self.num_class), dtype=np.int_)

        index = 0
        while 1:
            if index == self.fnum:
                index = 0
                random.shuffle(self.dataSet)

            for i in range(batch_size):
                fpath, labels = self.dataSet[index]
                index += 1

                img = cv2.imread(fpath, 1)
                img = processImg(img)

                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # 获取Lab图像

                img_L[i] = img_lab[:, :, :1].astype(np.float32)  # L空间
                img_ab[i] = img_lab[:, :, 1:].astype(np.float32)  # ab空间
                output_labels[i] = labels  # 类别标签

            # 归一化
            img_L = ImageProcessor.norm_L(img_L)
            img_ab = ImageProcessor.norm_ab(img_ab)

            yield ([img_L, img_L], [img_ab, output_labels])

    def train(self, epochs, batch_size):
        self.model.compile(optimizer='adam',  # adadelta -> adam
                           loss=['mse', 'categorical_crossentropy'],
                           loss_weights=[1, 1. / 300])

        """训练模型"""
        filePath = '/mnt/train.txt'
        root = '/mnt/'
        self.dataSet, self.num_class = DataGenerator.flowFromFile(filePath, root, batch_size)
        self.fnum = len(self.dataSet)

        batch_num = int(self.fnum / batch_size)  # 每个epoch训练轮数
        dataGenerator = self.dataIter(batch_size)

        saver = ModelSaver()
        self.model.fit(dataGenerator, epochs=epochs, steps_per_epoch=batch_num, callbacks=[saver])

        self.low_levelNet.save_weights('/mnt/lowNetWeights.h5')
        self.mid_levelNet.save_weights('/mnt/midNetWeights.h5')
        self.globalFeatNet.save_weights('/mnt/globalNetWeights.h5')
        self.colorizationNet.save_weights('/mnt/colorizeNetWeights.h5')
        self.classifyNet.save_weights('/mnt/classifyNetWeights.h5')

