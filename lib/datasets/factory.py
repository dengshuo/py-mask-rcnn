# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.tuzhen import tuzhen
# from datasets.cityscape import Cityscape
from datasets.tuzhen_hybrid import tuzhen2
from datasets.tuzhen_hybrid_mask_only import tuzhen_hybrid_mask_only
from datasets.tuzhen_hybrid_keypoints import tuzhen_hybrid_keypoints
from datasets.cityscape_pascal_voc import cityscape_pascal_voc
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

#Set up tuzhen_2017
for year in ['2017']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'tuzhen_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: tuzhen(split, year))

#Set up tuzhen_2017
for year in ['2018']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'tuzhen_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: tuzhen2(split, year))

#Set up tuzhen_2017
for year in ['2018']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'cityscape_pascal_voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: cityscape_pascal_voc(split, year))
        

#Set up tuzhen_2017
for year in ['2018']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'tuzhen_mask_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: tuzhen_hybrid_mask_only(split, year))


#Set up tuzhen_2017
for year in ['2018']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'tuzhen_keypoints_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: tuzhen_hybrid_keypoints(split, year))

# for split in ['train', 'test']:
#     name = 'cityscape_{}'.format(split)
#     __sets[name] = (lambda split=split, year=year: Cityscape(split,'/work1/dataset','/work1/dataset/cityscapes'))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
