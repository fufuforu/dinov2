


# handle coco
mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}



# handle lmot 

lmot_category2name = {
    1: 'person', # 1
    2: 'bicycle', # 2
    3: 'car', # 3
    4: 'motorcycle', # 4
    5: 'bus', # 6
    6: 'truck', # 8
}

lmot_category2label = {k: i for i, k in enumerate(lmot_category2name.keys())}
lmot_label2category = {v: k for k, v in lmot_category2label.items()}

# handle soda-d 

sodad_category2name = {
    1: 'people', # 1
    2: 'rider', # 2
    3: 'bicycle', # 3
    4: 'motor', # 4
    5: 'vehicle', # 5
    6: 'traffic-sign', # 6
    7: 'traffic-light', # 7
    8: 'traffic-camera', # 8
    9: 'warning-cone', # 9
}

sodad_category2label = {k: i for i, k in enumerate(lmot_category2name.keys())}
sodad_label2category = {v: k for k, v in lmot_category2label.items()}




LABEL2CATEGORY_DICT = {
    'coco': mscoco_label2category,
    'lmot': lmot_label2category,
    'sodad': sodad_label2category
}

CATEGORY2LABEL_DICT = {
    'coco': mscoco_category2label,
    'lmot': lmot_category2label,
    'sodad': sodad_category2label
}

CATEGORY2NAME_DICT = {
    'coco': mscoco_category2name,
    'lmot': lmot_category2name,
    'sodad': sodad_category2name
}