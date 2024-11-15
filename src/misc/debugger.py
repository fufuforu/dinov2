# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import math
import cv2
import os
import sys
import torch
import copy
from .instances import BatchInstances, Instances

class Debugger(object):
    def __init__(
            self,  
            show_pad=50, 
            show_max_size=800,
            show_images=[], # list of image names to show, if not provided, all images will be shown
            save_dir = 'output/debugger_vis', # the directory to save visualized images
            save_images=[], # list of image names to save, if not provided, all images will be saved
            pause=True, # pause for debugging
            num_categories=81,  # number of forground category pluse one background color
            theme='white'
        ):
        self.show_pad = show_pad
        self.show_max_size = show_max_size
        self.show_images = show_images
        self.save_images = save_images
        self.save_dir = save_dir
        self.pause = pause
        self.imgs = {}
        self.theme = theme

        self.num_categories = num_categories
        self._prepare_category_colors(self.num_categories)

        self.track_color = {}

    def reset(self):
        self.imgs = {}

    def set_category_names(self, category_names):
        self.category_names = category_names
        self._prepare_category_colors(len(self.category_names))

    def _prepare_category_colors(self, count):
        colors = [(color_list[i]).astype(np.uint8) for i in range(count)]
        while len(colors) < count:
            colors = colors + colors[:min(len(colors), count - len(colors))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)

    def preprocess_image(self, img, img_type='RGB'):
        if isinstance(img, torch.Tensor):
            img = img.to(torch.device("cpu")).numpy()
        # img = img * 255
        if img.mean() <= 20:
            img = img * 255
        
        # # amply for dark images
        # scale = 127 / img.mean()
        # img = img * scale

        assert len(img.shape) == 3
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
        img = np.asarray(img, dtype=np.uint8)
        if img_type == 'RGB':
            img = img[:, :, ::-1]
        return img

    def add_img(self, img, img_id='default', img_type='RGB', revert_color=False):
            
        if revert_color:
            img = 255 - img
        
        pad = self.show_pad
        if pad != 0:
            img = cv2.copyMakeBorder(img.copy(), pad, pad, pad, pad, cv2.BORDER_CONSTANT)

        self.imgs[img_id] = img.copy()

    def add_blend_img(self, back, fore, img_id='blend', trans=0.8):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    
    def gen_colormap(self, img):
        """
        img: the predicted heatmap. 3D tensor or array
        """
        if isinstance(img, torch.Tensor):
            img = img.to(torch.device('cpu')).numpy()
        img = img.copy()
        # ignore region
        # import pdb; pdb.set_trace()
        img[img == 1] = 0.5
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        #TODO
        colors = np.array(self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        # colors = np.array([0, 0, 255], dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3) # red
        if self.theme == 'white':
            colors = 255 - colors

        color_map = (img * colors).max(axis=2).astype(np.uint8)
        return color_map

    def _get_rand_color(self):
        c = ((np.random.random((3)) * 0.6 + 0.2) * 255).astype(np.int32).tolist()
        return c

    def add_arrow(self, st, ed, img_id, c=(255, 0, 255)):
        cv2.arrowedLine(
            self.imgs[img_id], (int(st[0]), int(st[1])),
            (int(ed[0] + st[0]), int(ed[1] + st[1])), c, 2,
            line_type=cv2.LINE_AA, tipLength=0.3)

    def add_text(self, text, img_id, coord=(5, 5),  c=(255, 0, 255)):
        pad = self.show_pad
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_size = 1
        cat_size = cv2.getTextSize(text, font, font_size, thickness)[0]
        cv2.putText(self.imgs[img_id], text, (coord[0]+pad, coord[1]+pad+cat_size[1]+thickness), 
                    font, font_size, c, thickness=thickness, lineType=cv2.LINE_AA)


    def add_bbox(self, bbox, cat, conf=1, track_id=None, show_txt=True, img_id='default'):
        pad = self.show_pad
        bbox = np.array(bbox, dtype=np.int32) + pad

        cat = int(cat) if cat is not None else 0
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        
        if hasattr(self, 'category_names'):
            cat = self.category_names[cat][:4]
        txt = '{}:{:.1f}'.format(cat, conf) if conf is not None else '{}'.format(cat)
        thickness = 3 #2
        fontsize = 1 # 0.5
        if track_id is not None:
            track_id = int(track_id)
            if not (track_id % 1000 in self.track_color):
                self.track_color[track_id % 1000] = self._get_rand_color()
            c = self.track_color[track_id % 1000]
            txt = '{}:{}'.format(cat, track_id) if hasattr(self, 'category_names') else '{}'.format(track_id)
        cv2.rectangle(self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, thickness)

        if show_txt:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
            cv2.rectangle(self.imgs[img_id],
                            (bbox[0], bbox[1] - cat_size[1] - thickness),
                            (bbox[0] + cat_size[0], bbox[1]), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - thickness - 1),
                        font, fontsize, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
 

    def add_image_with_heatmap(self, img, heatmap, img_id='default'):
        img = self.preprocess_image(img)
        hm = self.gen_colormap(heatmap)
        self.add_blend_img(back=img, fore=hm, img_id=img_id)

    def add_image_with_bbox(self, img, meta_data, img_id='default', box_type='', img_type='RGB'):
        img = self.preprocess_image(img=img, img_type=img_type)
        im_h, im_w = img.shape[:2]
        self.add_img(img, img_id)
        
        if isinstance(meta_data, dict):
            boxes = meta_data['boxes']
            labels = meta_data.get('labels', None)
            obj_ids = meta_data.get('obj_ids', None)
            scores = meta_data.get('scores', None)
        elif isinstance(meta_data, torch.Tensor):
            boxes = meta_data[:,:4]
            labels = None
            obj_ids = None
            scores = None
        elif isinstance(meta_data, (BatchInstances, Instances)): # BatchInstances or Instances
            boxes = meta_data.boxes.data
            labels = getattr(meta_data, 'labels', None)
            obj_ids = getattr(meta_data, 'obj_ids', None)
            scores = getattr(meta_data, 'scores', None)
            if len(boxes.shape) == 3: # remove batch
                boxes = boxes[0] if boxes is not None else None
                labels = labels[0] if labels is not None else None
                obj_ids = obj_ids[0] if obj_ids is not None else None
                scores = scores[0] if scores is not None else None 
        else:
            raise NotImplementedError('Unknown type of meta_data: {}'.format(type(meta_data)))


        if len(boxes) == 0:
            return

        # # import pdb; pdb.set_trace()
        # if 'xyxy' in box_type:
        #     pass
        # elif 'cxcywh' in box_type:
        #     boxes_ = copy.deepcopy(getattr(boxes, 'data', boxes))
        #     boxes_[0] = boxes[0] - boxes[2]/2
        #     boxes_[1] = boxes[1] - boxes[3]/2
        #     boxes_[2] = boxes[0] + boxes[2]/2
        #     boxes_[3] = boxes[1] + boxes[3]/2
        #     boxes = boxes_
        # else:
        #     raise NotImplementedError
        
        # if 'norm' in box_type:
        #     boxes[0::2] *= im_w
        #     boxes[1::2] *= im_h

        for i in range(len(boxes)):
            xyxy = boxes[i]
            label = labels[i] if labels is not None else None
            object_id = obj_ids[i] if obj_ids is not None else None
            score = scores[i] if scores is not None else None
            
            if 'xyxy' in box_type:
                pass 
            elif 'cxcywh' in box_type:
                xyxy_ = copy.deepcopy(xyxy)
                xyxy_[0] = xyxy[0] - xyxy[2]/2
                xyxy_[1] = xyxy[1] - xyxy[3]/2
                xyxy_[2] = xyxy[0] + xyxy[2]/2
                xyxy_[3] = xyxy[1] + xyxy[3]/2
                xyxy = xyxy_
            else:
                raise NotImplementedError

            if 'norm' in box_type:
                xyxy = [xyxy[0]*im_w, xyxy[1]*im_h, xyxy[2]*im_w, xyxy[3]*im_h]

            self.add_bbox(xyxy, label, score, object_id, img_id=img_id)


    def show_all_imgs(self, Time=0, sup_title=None):

        def _get_size(height, width, min_size, max_size):
            """
            Given the image height and size, return the obtained size and 
            scale to resize the image
            """
            size = min_size
            if max_size is not None:
                min_original_size = float(min((width, height)))
                max_original_size = float(max((width, height)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (width <= height and width == size) or (height <= width and height == size):
                return (height, width)

            if width < height:
                ow = size
                oh = int(size * height / width)
            else:
                oh = size
                ow = int(size * width / height)

            return oh, ow

        def _resize_image(image):
            height, width = image.shape[0], image.shape[1]
            if max(height, width) > self.show_max_size:
                min_size = min(height, width, 0.5 * self.show_max_size)
                height_s, width_s = _get_size(height, width, min_size, self.show_max_size)
                image = cv2.resize(image, (int(width_s), int(height_s)))
            return image

        if 1:
            for i, v in self.imgs.items():
                if i in list(self.show_images) or len(self.show_images) == 0:
                    imshow = _resize_image(v)
                    im_title = '{}'.format(i)
                    cv2.imshow(im_title, imshow)
            if cv2.waitKey(0 if self.pause else 1) == 27: # press esc to quit
                # cv2.destroyAllWindows()
                sys.exit(0)
        else:
            keys = [k for k in self.imgs.keys()]
            if len(self.show_images):
                keys = [k for k in keys if k in list(self.show_images)]
            nImgs = len(keys)
            nCols = math.ceil(math.sqrt(nImgs))
            nRows = math.ceil(nImgs/nCols)
            assert nCols * nRows >= nImgs
            # fig = plt.figure(figsize=(nCols * 10, nRows * 10))
            plt.cla()
            # for i, (k, v) in enumerate(self.imgs.items()):
            for i in range(len(keys)):
                k = keys[i]
                v = self.imgs[k]
                plt.subplot(nRows, nCols, i+1)
                if len(v.shape) == 3:
                    plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(v)
                plt.title(k)
                plt.axis('off')
            # plt.show()
            if sup_title is not None:
                plt.suptitle(sup_title)
            if not self.pause: # 
                plt.pause(1e-16)
            else:
                plt.show()

    def save_all_imgs(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        for i, v in self.imgs.items():
            # import pdb; pdb.set_trace()
            if i in list(self.save_images) or len(self.save_images) == 0:
                i_path = os.path.join(save_dir, '{}.jpg'.format(i))
                cv2.imwrite(i_path, v)
                print('saved to {}'.format(i_path))



color_list = np.array(
    [0.850, 0.325, 0.098,
     1.000, 1.000, 1.000, 
     0.929, 0.694, 0.125,
     0.494, 0.184, 0.556,
     0.466, 0.674, 0.188,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500,
     0.333, 0.000, 0.500,
     0.333, 0.333, 0.500,
     0.333, 0.667, 0.500,
     0.333, 1.000, 0.500,
     0.667, 0.000, 0.500,
     0.667, 0.333, 0.500,
     0.667, 0.667, 0.500,
     0.667, 1.000, 0.500,
     1.000, 0.000, 0.500,
     1.000, 0.333, 0.500,
     1.000, 0.667, 0.500,
     1.000, 1.000, 0.500,
     0.000, 0.333, 1.000,
     0.000, 0.667, 1.000,
     0.000, 1.000, 1.000,
     0.333, 0.000, 1.000,
     0.333, 0.333, 1.000,
     0.333, 0.667, 1.000,
     0.333, 1.000, 1.000,
     0.667, 0.000, 1.000,
     0.667, 0.333, 1.000,
     0.667, 0.667, 1.000,
     0.667, 1.000, 1.000,
     1.000, 0.000, 1.000,
     1.000, 0.333, 1.000,
     1.000, 0.667, 1.000,
     0.167, 0.000, 0.000,
     0.333, 0.000, 0.000,
     0.500, 0.000, 0.000,
     0.667, 0.000, 0.000,
     0.833, 0.000, 0.000,
     1.000, 0.000, 0.000,
     0.000, 0.167, 0.000,
     0.000, 0.333, 0.000,
     0.000, 0.500, 0.000,
     0.000, 0.667, 0.000,
     0.000, 0.833, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 0.000,
     0.000, 0.000, 0.167,
     0.000, 0.000, 0.333,
     0.000, 0.000, 0.500,
     0.000, 0.000, 0.667,
     0.000, 0.000, 0.833,
     0.000, 0.000, 1.000,
     0.333, 0.000, 0.500,
     0.143, 0.143, 0.143,
     0.286, 0.286, 0.286,
     0.429, 0.429, 0.429,
     0.571, 0.571, 0.571,
     0.714, 0.714, 0.714,
     0.857, 0.857, 0.857,
     0.000, 0.447, 0.741,
     0.50, 0.5, 0
     ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
