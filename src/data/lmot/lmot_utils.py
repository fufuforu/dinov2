import cv2
import numpy as np
import torch


class RawImageReader(object):
    def __init__(self, n_bit=None):
        self.capture_bit = 16
        self.capture_bit_max = 2 ** self.capture_bit

        self.n_bit = n_bit if n_bit is not None else self.capture_bit
        
        assert self.n_bit >= 1 and self.n_bit <= self.capture_bit
        self.n_bit_max = 2 ** self.n_bit
        self.n_bit_scale = 1 / (2 ** (self.capture_bit - self.n_bit))


    def __call__(self, img_file, black_level, numpy_or_tensor='tensor'):
        # cv2.ocl.setUseOpenCL(False)
        # cv2.setNumThreads(0)
        img = cv2.imread(img_file, -1)

        img = img * self.n_bit_scale
        img = img.round().clip(0, self.n_bit_max-1)

        # BayerRG -> RGB
        black_level = round(black_level * self.n_bit_scale)
        img = np.clip(img.astype(np.float32) - black_level, 0, self.n_bit_max-1).astype(np.uint16) # the images in LMOT are captured with 16 bits
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB).astype(np.float32)
        
        # AWB
        mean_r = img[:, :, 0].mean()
        mean_g = img[:, :, 1].mean()
        mean_b = img[:, :, 2].mean()
        img[:, :, 0] *= mean_g / mean_r
        img[:, :, 2] *= mean_g / mean_b

        # normalize to [0, 255]
        img = np.clip(img, 0, self.n_bit_max-1) / (self.n_bit_max-1) * 255  # RGB in [0, 255] 
        
        if numpy_or_tensor == 'tensor':
            img = torch.Tensor(img).permute(2, 0, 1) # [h, w, 3] -> [3, h, w]
        
        return img 