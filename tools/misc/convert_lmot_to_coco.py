import os
import numpy as np
import json
import glob
import cv2
import argparse

CWD = os.path.dirname(__file__)


def save_ann(data, save_path):
    print('==> saved in {}'.format(save_path))
    print('==> {} images, {} instances'.format(len(data['images']), len(data['annotations'])))
    json.dump(data, open(save_path, 'w'), indent=4)


def convert_lmot(args):
    if not os.path.exists(os.path.join(args.root, 'LMOT')):
        print(' ==== LMOT not found ====')
        return

    
    splits = ['train', 'val', 'test', 'real']
    for split in splits:
        if split in ['train', 'val', 'test']:
            img_types = ['img_dark', 'img_dark_ns_isp', 'img_light', 'img_light_isp']
        else:
            img_types = ['img_real', 'img_real_ns_isp']

        for img_type in img_types:

            img_id = 0
            instance_id = 0
            track_count = 0

            data_path = os.path.join(args.root, 'LMOT', 'images', split)
            out_path = os.path.join(args.root, 'LMOT', 'annotations', '{}_{}.json'.format(split, img_type))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            out = {
                'sequences': [],
                'images': [], 
                'annotations':[],
                'categories': [
                    {
                        'id': 1,
                        'name': 'person',
                    },
                    {
                        'id': 2,
                        'name': 'bicycle',
                    },
                    {
                        'id': 3,
                        'name': 'car',
                    },
                    {
                        'id': 4,
                        'name': 'motorcycle',
                    },
                    {
                        'id': 5,
                        'name': 'bus',
                    },
                    {
                        'id': 6,
                        'name': 'truck',
                    },
                ]
            }
            
            seqs = [s for s in os.listdir(data_path)]
            
            for seq_i in range(len(seqs)):
                seq = seqs[seq_i]
                seq_info = open(os.path.join(data_path, seq, 'seqinfo.ini')).read().split('\n')
                seq_info_d = {}
                for s in seq_info:
                    if '=' in s:
                        s = s.replace(' ', '').split('=')
                        seq_info_d[s[0]] = s[1]

                seq_info = {
                    'id': seq_i,
                    'width': int(seq_info_d['imwidth']),
                    'height': int(seq_info_d['imheight']),
                    'black_level': int(seq_info_d['black_level']),
                    'num_frames': int(seq_info_d['seqlength']),
                    'sequence_name': seq,
                    'image_ids': [], # store the image id for this video, in ordered,
                }
                
                gt_txt = os.path.join(data_path, seq, 'gt', 'gt.txt')
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',') # fid, tid, x, y, w, h, mark, category_id, vis 

                for fid in range(1, seq_info['num_frames']+1):
                    img_id += 1
                    
                    # the info of image
                    if img_type.endswith('_isp'):
                        img_ext = '.png'
                    else:
                        img_ext = '.tiff'
                    # frame_path = os.path.join('LMOT', 'images', split, seq, img_type, '{:06d}{}'.format(int(fid), img_ext))
                    frame_path = os.path.join(split, seq, img_type, '{:06d}{}'.format(int(fid), img_ext))

                    img_info = {
                        'sequence_id': seq_i,
                        'id': img_id,
                        'file_name': frame_path,
                        'frame_id': int(fid), 
                        'width': seq_info['width'],
                        'height': seq_info['height'],
                        'black_level': seq_info['black_level'],
                    }

                    seq_info['image_ids'].append(img_id)
                    out['images'].append(img_info)
                    
                    # the info of each annotations
                    gt_ = gt[gt[:, 0]==fid]
                    for fid, tid, x, y, w, h, mark, category_id, visbility in gt_:
                        if mark == 0:
                            continue
                        
                        instance_id += 1

                        ann_info = {
                            'id': instance_id,
                            'image_id': img_id,
                            'category_id': category_id,
                            'iscrowd': 0, # not crowded
                            'area': w*h,
                            'bbox': [x, y, w, h],
                            'width': seq_info['width'],
                            'height': seq_info['height'], 

                            # for tracking
                            'visibility': visbility, # binary, 0 means unvisible
                            'track_id': tid + track_count
                        }
                        out['annotations'].append(ann_info)
            
                track_count = track_count + gt[:, 1].max()

                out['sequences'].append(seq_info)

            save_ann(out, out_path)


def ArgParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='dataset')
    args = parser.parse_args()
    # args.cwd = os.path.abspath(os.path.dirname(__file__))
    # if args.root == '':
    #     args.root = os.path.join(args.cwd, '../../DATASET')
    return args

if __name__ == '__main__':
    args = ArgParse()
    convert_lmot(args)
