# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import cv2
import string
import random

DEBUG = True

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._count = 0
        self._fg_num =0
        self._bg_num =0

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)

        # mask_targets
        top[5].reshape(1,self._num_classes,cfg.MASK_SIZE,cfg.MASK_SIZE)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        image = bottom[0].data
        flipped = bottom[1].data
        all_rois = bottom[2].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[3].data

        #gt_masks;
        gt_masks = bottom[4].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights,mask_targets = _sample_rois(image,flipped,
            all_rois, gt_boxes, gt_masks,fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        #mask_targets
        top[5].reshape(*mask_targets.shape)
        top[5].data[...] = mask_targets

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        start = int(start)
        end = int(end)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(image,flipped,all_rois,gt_boxes,gt_masks,fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    #all_mask_targets = np.zeros((len(labels),81,cfg.MASK_SIZE,cfg.MASK_SIZE), dtype=np.float32)
    gt_boxes_ = gt_boxes[gt_assignment, :4]
    # for i in np.arange(len(labels)):
    #     #print i,labels[i]
    #     for j in np.arange(num_classes):
    #         if(int(labels[i]) == 0):
    #             all_mask_targets[i, 0, :, :] = -1
    #         if(j == int(labels[i])):
    #             all_mask_targets[i, j, :, :] = gt_masks[gt_assignment[i]]
    #         else:
    #             all_mask_targets[i, j, :, :] = -1


    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    fg_rois_per_this_image = int(fg_rois_per_this_image)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    original_labels = labels
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    #mask_targets = all_mask_targets[keep_inds,:,:,:]
    bboxes = gt_boxes_[keep_inds, :4]
    keep_assingment_inds = gt_assignment[keep_inds]
    mask_targets = np.zeros((len(labels),81,cfg.MASK_SIZE,cfg.MASK_SIZE), dtype=np.float32)
    mask_targets[:,:,:,:] = -1
    #mask_assignment = gt_assignment[keep_inds]
    # for i in np.arange(len(labels)):
    #     #print i,labels[i]
    #     for j in np.arange(num_classes):
    #         if int(labels[i]) == 0:
    #             mask_targets[i, 0, :, :] = -1
    #             if original_labels[i] != labels[i]:
    #                 mask_targets[i, original_labels[i], :, :] = -1

    for i in np.arange(len(labels)):
        if int(labels[i]) == 0:
            mask_targets[i, 0, :, :] = -1
            if int(original_labels[i]) != int(labels[i]):
                mask_targets[i, original_labels[i], :, :] = -1
        elif int(labels[i]) == int(original_labels[i]):
            x1 = rois[i, 1]
            y1 = rois[i, 2]
            x2 = rois[i, 3]
            y2 = rois[i, 4]
            j = int(labels[i])
            # print x1,y1,x2,y2
            # gt_mask = mask_targets[i,int(original_labels[i]),:,:]
            gt_mask = gt_masks[keep_assingment_inds[i],:,:]
            bbox = bboxes[i, :4]
            restored_gt_mask = cv2.resize(gt_mask, (int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])),
                                          interpolation=cv2.INTER_CUBIC)
            if cfg.DEBUG:
                image_ori = image.copy()
                image_ori[0,int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])][restored_gt_mask<0.5]=(0,0,0)
                file_name = string.join(random.sample(['a','b','c','d','e','f','g','h','i','j','k','l','m','n'], 6)).replace(" ","")
                image_ = image_ori[0,:,:,:]
                #cv2.rectangle(image_,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),1)
                cv2.rectangle(image_,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),1)
                cv2.putText(image_,cfg.COCOCLASS[j],(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                #cv2.putText()
                #cv2.imwrite(cfg.ROOT_DIR+"/"+file_name+".jpg",image_)
                #print image_
                if flipped[0] == 1:
                    cv2.imwrite(cfg.ROOT_DIR+"/test/gt-f/"+file_name + "__gt.jpg", image_)
                else:
                    cv2.imwrite(cfg.ROOT_DIR+"/test/gt/"+file_name + "__gt.jpg", image_)
            # all_mask_targets[i, j, :, :] = -1
            
            mask = np.zeros((np.round(y2 - y1).astype(np.int), np.round(x2 - x1).astype(np.int)), dtype=np.float32)
            mask[:, :] = 0
            # width = mask.shape[1]
            # height = mask.shape[0]
            if (int(bbox[2]) - int(bbox[0])) <= 0 or (int(bbox[3]) - int(bbox[1])) <= 0 or bbox[2] < 0 or bbox[
                0] < 0 or bbox[1] < 0 or bbox[3] < 0:
                # resized_mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_CUBIC)
                mask_targets[i, j, :, :] = 0
                continue

            if (np.round(y2 - y1).astype(np.int) <= 0) or (np.round(x2 - x1).astype(np.int)) <= 0:
                mask_targets[i, j, :, :] = 0
                continue

            # restored_gt_mask = cv2.resize(gt_mask, (int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])),
            #                               interpolation=cv2.INTER_LINEAR)
            i_top_left_x = np.round(np.max((x1, bbox[0]))).astype(np.int)
            i_bottom_right_x = np.round(np.min((x2, bbox[2]))).astype(np.int)
            i_top_left_y = np.round(np.max((y1, bbox[1]))).astype(np.int)
            i_bottom_right_y = np.round(np.min((y2, bbox[3]))).astype(np.int)

            if i_top_left_x == np.round(x1).astype(np.int):
                roi_start_x = 0
                roi_end_x = i_bottom_right_x - i_top_left_x
                # gt_start_x = bbox[2]-(i_bottom_right_x-i_top_left_x)
                gt_start_x = np.round(i_top_left_x - bbox[0]).astype(np.int)
                gt_end_x = gt_start_x + (i_bottom_right_x - i_top_left_x)
            else:
                roi_start_x = np.round(i_top_left_x - x1).astype(np.int)
                roi_end_x = roi_start_x + i_bottom_right_x - i_top_left_x
                gt_start_x = 0
                gt_end_x = (i_bottom_right_x - i_top_left_x)

            if i_top_left_y == np.round(y1).astype(np.int):
                roi_start_y = 0
                roi_end_y = i_bottom_right_y - i_top_left_y
                gt_start_y = np.round(i_top_left_y - bbox[1]).astype(np.int)
                gt_end_y = gt_start_y + (i_bottom_right_y - i_top_left_y)
            else:
                roi_start_y = np.round(i_top_left_y - y1).astype(np.int)
                roi_end_y = roi_start_y + i_bottom_right_y - i_top_left_y
                gt_start_y = 0
                gt_end_y = i_bottom_right_y - i_top_left_y

            if roi_end_x == roi_start_x:
                roi_end_x += 1
            if roi_start_y == roi_end_y:
                roi_end_y += 1

            if gt_start_x == gt_end_x:
                gt_end_x += 1
            if gt_start_y == gt_end_y:
                gt_end_y += 1

            if roi_end_y - roi_start_y > gt_end_y - gt_start_y:
                roi_end_y -= 1
            elif roi_end_y - roi_start_y < gt_end_y - gt_start_y:
                gt_end_y -= 1

            if roi_end_x - roi_start_x > gt_end_x - gt_start_x:
                roi_end_x -= 1
            elif roi_end_x - roi_start_x < gt_end_x - gt_start_x:
                gt_end_x -= 1

            # if roi_end_y-roi_start_y > gt_end_y-gt_start_y:
            #     roi_end_y -=1
            # elif roi_end_y-roi_start_y< gt_end_y-gt_start_y:
            #     gt_end_y -= 1
            #
            # if roi_end_x-roi_start_x > gt_end_x-gt_start_x:
            #     roi_end_x -= 1
            # elif roi_end_x - roi_start_x < gt_end_x - gt_start_x:
            #     gt_end_x -= 1

            if gt_end_x == restored_gt_mask.shape[1] + 1:
                gt_end_x -= 1
                roi_end_x -= 1

            if gt_end_y == restored_gt_mask.shape[0] + 1:
                gt_end_y -= 1
                roi_end_y -= 1

            if roi_end_x == mask.shape[1] + 1:
                gt_end_x -= 1
                roi_end_x -= 1

            if roi_end_y == mask.shape[0] + 1:
                gt_end_y -= 1
                roi_end_y -= 1

            if i_bottom_right_y <= i_top_left_y:
                # resized_mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_LINEAR)
                mask_targets[i, j, :, :] = 0
                continue
            if i_bottom_right_x <= i_top_left_x:
                # resized_mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_LINEAR)
                mask_targets[i, j, :, :] = 0
                continue
            # print roi_start_y, roi_end_y, roi_start_x, roi_end_x, gt_start_y, gt_end_y, gt_start_x, gt_end_x,restored_gt_mask.shape,mask.shape,i_bottom_right_y,i_top_left_y
            mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = restored_gt_mask[gt_start_y:gt_end_y,gt_start_x:gt_end_x]

            # start_h = np.round(np.max((1, y1))).astype(np.int)
            # end_h = np.round(np.min((height, y2))).astype(np.int)
            # start_w = np.round(np.max((1, x1))).astype(np.int)
            # end_w = np.round(np.min((width, x2))).astype(np.int)
            # if start_h==end_h:
            #     end_h +=1
            # if start_w == end_w:
            #     end_w +=1
            # cropped_mask = mask[start_h:end_h, start_w:end_w]
            # print '\nstart_h:end_h, start_w:end_w',start_h,end_h, start_w,end_w
            resized_mask = cv2.resize(mask, (cfg.MASK_SIZE,cfg.MASK_SIZE), interpolation=cv2.INTER_CUBIC)
            mask_targets[i, j, :, :] = resized_mask
            #print 'flipped:',flipped[0]
            if cfg.DEBUG == True:
                image_ori = image.copy()
                image_ori[0,int(y1):mask.shape[0]+int(y1),int(x1):mask.shape[1]+int(x1)][mask<0.5]=(0,0,0)
                file_name = string.join(random.sample(['a','b','c','d','e','f','g','h','i','j','k','l','m','n'], 6)).replace(" ","")
                image_ = image_ori[0,:,:,:]
                cv2.rectangle(image_,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),1)
                cv2.rectangle(image_,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),1)
                cv2.putText(image_,cfg.COCOCLASS[j],(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                #cv2.putText()
                #cv2.imwrite(cfg.ROOT_DIR+"/"+file_name+".jpg",image_)
                #print image_
                if  flipped[0] == 1:
                    cv2.imwrite(cfg.ROOT_DIR+"/test/roi-f/"+file_name + ".jpg", image_)
                else:
                    cv2.imwrite(cfg.ROOT_DIR+"/test/roi/"+file_name + ".jpg", image_)

        else:
            mask_targets[i, j, :, :] = -1

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights,mask_targets
