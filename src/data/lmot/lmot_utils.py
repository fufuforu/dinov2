import cv2
import numpy as np
import torch


class RawImageReader(object):
    def __init__(self, is_dark, n_bit=None):
        
        self.is_dark = is_dark

        self.capture_bit = 16
        self.capture_bit_max = 2 ** self.capture_bit

        self.n_bit = n_bit if n_bit is not None else self.capture_bit
        
        assert self.n_bit >= 1 and self.n_bit <= self.capture_bit
        self.n_bit_max = 2 ** self.n_bit
        self.n_bit_scale = 1 / (2 ** (self.capture_bit - self.n_bit))
        # self.n_bit_scale = 1 / (self.capture_bit_max - 1) * self.n_bit_max

    def __call__(self, img_file, black_level, numpy_or_tensor='tensor'):
        # return self.read_v1(img_file=img_file, black_level=black_level, numpy_or_tensor=numpy_or_tensor)
        return self.read_v2(img_file=img_file, black_level=black_level, numpy_or_tensor=numpy_or_tensor)

    def read_v1(self, img_file, black_level, numpy_or_tensor='tensor'):
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

        # cv2.imwrite('test.png', img.astype(np.uint8)[:,:,::-1])
        # import pdb; pdb.set_trace()
        
        if numpy_or_tensor == 'tensor':
            img = torch.Tensor(img).permute(2, 0, 1) # [h, w, 3] -> [3, h, w]
        
        return img 
    
    def read_v2(self, img_file, black_level, numpy_or_tensor='tensor'):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        img = cv2.imread(img_file, -1)
        assert img is not None

        if self.n_bit is not None:
            img = img * self.n_bit_scale
            img = img.round().clip(0, self.n_bit_max)
            img = img / self.n_bit_scale

        # black level
        img = np.clip(img.astype(np.float32) - black_level, 0, self.capture_bit_max - 1).astype(np.uint16)
        
        # demosaic
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB).astype(np.float32)
        
        # awb
        mean_r = img[:, :, 0].mean()
        mean_g = img[:, :, 1].mean()
        mean_b = img[:, :, 2].mean()
        img[:, :, 0] *= mean_g / mean_r
        img[:, :, 2] *= mean_g / mean_b
        
        # Scale
        # import pdb; pdb.set_trace()
        if self.is_dark:
            img = img / img.mean() * 0.2
        else:
            img = img / (self.capture_bit_max - 1)
        
        # Gamma
        img = np.power(img, 1/1.8)
        img = img * 255
        
        if numpy_or_tensor == 'tensor':
            img = torch.Tensor(img).permute(2, 0, 1) # [h, w, 3] -> [3, h, w]

        return img



if __name__ == '__main__':
    img_path = 'dataset/LMOT/images/train/LMOT-02/img_dark/000001.tiff'
    black_level = 394

    reader = RawImageReader(is_dark=True)
    img = reader(img_path, black_level)
    img2 = reader.read_v2(img_path, black_level)
    img3 = reader.read_v3(img_path, black_level)

    # import pdb; pdb.set_trace()