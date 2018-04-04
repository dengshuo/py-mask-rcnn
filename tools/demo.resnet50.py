#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import string
import random



CLASSES = ('background',  # class zero
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

NETS = {'vgg16': ('VGG16',
                  'vgg16_mask_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg1024': ('VGG_CNN_M_1024',
                  'vgg_cnn_m_1024_faster_rcnn_final.caffemodel'),
        'resnet50': ('ResNet50',
            'resnet_50_mask_rcnn_final.caffemodel'),
        }


def vis_detections(im, class_name,cls_ind, dets, masks,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print 'cls_ind',cls_ind,'masks shape:',masks.shape,'i:',i
        mask = masks[i,cls_ind,:,:]
        print '\nmask:',mask
        # mask_restore = cv2.resize(mask, (int(bbox[2]) - int(bbox[0]),int(bbox[3]) - int(bbox[1])), interpolation=cv2.INTER_NEAREST)
        #mask_restore = np.round(mask_restore)
        mask_restore = cv2.resize(mask, (int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])),
                                  interpolation=cv2.INTER_CUBIC)
        print '\nmask restore:',mask_restore,'maks_restore shape:',mask_restore.shape
        #print '\nmask restore',mask_restore
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])][mask_restore<0] =(0,0,0)

    file_name = string.join(random.sample(['a','b','c','d','e','f','g','h','i','j','k','l','m','n'], 6)).replace(" ","")
    #image_ = image[0,:,:,:]
    #cv2.rectangle(image_,(int(x1),int(y1)),(int(x2),int(y2)),blue,1)
    #cv2.rectangle(image_,(int(bbox[0]),int(bbox[1],(int(bbox[2]),int(bbox[3])),red,1)
    cv2.imwrite(cfg.ROOT_DIR+"/"+file_name + ".jpg", im)
    # ax.imshow(im, aspect='equal')
    # for i in inds:
    #     bbox = dets[i, :4]
    #     score = dets[i, -1]

    #     ax.add_patch(
    #         plt.Rectangle((bbox[0], bbox[1]),
    #                       bbox[2] - bbox[0],
    #                       bbox[3] - bbox[1], fill=False,
    #                       edgecolor='red', linewidth=3.5)
    #         )
    #     ax.text(bbox[0], bbox[1] - 2,
    #             '{:s} {:.3f}'.format(class_name, score),
    #             bbox=dict(facecolor='blue', alpha=0.5),
    #             fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes,pred_masks = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        masks = pred_masks[keep,:,:,:]
        vis_detections(im, cls, cls_ind,dets,masks,thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='resnet50')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'mask_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'mask_rcnn_models',
                              NETS[args.demo_net][1])

    print 'prototxt:',prototxt,'caffemodel:',caffemodel

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _,_= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg','COCO_train2014_000000000030.jpg']
    # im_names = ['COCO_train2014_000000000030.jpg']
    # im_names = ['COCO_test2014_000000000014.jpg','COCO_test2014_000000000016.jpg',
    #             'COCO_test2014_000000000027.jpg','COCO_test2014_000000000057.jpg']

    # im_names = ['COCO_train2014_000000000092.jpg','COCO_train2014_000000000094.jpg',
    #             'COCO_train2014_000000000109.jpg','COCO_train2014_000000000110.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
