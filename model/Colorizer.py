from .ColorizationModel import ColorizationNet
import cv2
import os.path
import numpy as np

class Colorizer(ColorizationNet):
    def __init__(self, modelDir):
        super().__init__()

        self.low_levelNet.load_weights(os.path.join(modelDir, 'lowNetWeights.h5'))
        self.mid_levelNet.load_weights(os.path.join(modelDir, 'midNetWeights.h5'))
        self.globalFeatNet.load_weights(os.path.join(modelDir, 'globalNetWeights.h5'))
        self.colorizationNet.load_weights(os.path.join(modelDir, 'colorizeNetWeights.h5'))

    def colorizeImg(self, fpath):
        def resizeImg(img):
            height, width, channel = img.shape

            tmp = height % 8
            height = height - (tmp if tmp <= 4 else tmp - 8)
            tmp = width % 8
            width = width - (tmp if tmp <= 4 else tmp - 8)

            img = cv2.resize(img, (width, height))

            return img

        img_gray = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), 0)  # 避免路径中包含中文时报错
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_gray = resizeImg(img_gray)

        img_bgr = self.colorize(img_gray)
        return img_bgr

