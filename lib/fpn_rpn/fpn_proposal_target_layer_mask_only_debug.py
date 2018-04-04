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


DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)
        top[2].reshape(1, 5)
        top[3].reshape(1, 5) 
        # labels
        top[4].reshape(4, 1)
        # bbox_targets
        top[5].reshape(4, self._num_classes * 4)
        # bbox_inside_weights
        top[6].reshape(4, self._num_classes * 4)
        # bbox_outside_weights
        top[7].reshape(4, self._num_classes * 4)

        # mask_targets
        top[8].reshape(4,self._num_classes-1,cfg.MASK_SIZE,cfg.MASK_SIZE)


    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        #gt_masks;
        gt_masks = bottom[2].data

        image_data = bottom[3].data

        image = bottom[4].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # print all_rois.shape,gt_boxes.shape
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
        labels, rois, gt_boxes,bbox_targets, bbox_inside_weights,mask_targets,rois_layers = _sample_rois(
            all_rois, gt_boxes, gt_masks,fg_rois_per_image,
            rois_per_image, self._num_classes,image_data)

        # permuted_lables, permuted_rois, bbox_targets, bbox_inside_weights,permuted_mask_targets,rois_layers = _sample_rois(
        #     all_rois, gt_boxes, gt_masks,fg_rois_per_image,
        #     rois_per_image, self._num_classes,image_data)
        # print rois,gt_boxes
        layer_rois = np.zeros((len(labels),5), dtype=np.float32)
        count =0
        for i in xrange(4):
            count_layer_i = rois_layers[i].shape[0]
            if count_layer_i != 0 :
                layer_rois[count:count_layer_i+count] = rois_layers[i]
                count += count_layer_i

        print 'len lables',len(labels)
        for i in np.arange(len(labels)):
            if int(labels[i]) != 0:
                x1 = np.round(rois[i, 1]).astype(np.int)
                y1 = np.round(rois[i, 2]).astype(np.int)
                x2 = np.round(rois[i, 3]).astype(np.int)
                y2 = np.round(rois[i, 4]).astype(np.int)
                # print 'x1,y1,x2,y2:',x1,y1,x2,y2
                # print 'layer x1,y1,x2,y2',layer_rois[i][1],layer_rois[i][2],layer_rois[i][3],layer_rois[i][4]
                j = int(labels[i])
                h = y2-y1
                w = x2-x1
                # print h,w
                restored_mask = cv2.resize(mask_targets[i,j-1,:,:], (w,h), interpolation=cv2.INTER_CUBIC)
                restored_mask = np.round(restored_mask)
                # print restored_mask.shape
                # image_copy = image.copy()
                # image_copy[y1:y2,x1:x2][restored_mask<0.5]=(0,0,0)
                
                # image_ = image_ori[0,:,:,:]
                cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),1)
                cv2.rectangle(image,(int(gt_boxes[i][0]),int(gt_boxes[i][1])),(int(gt_boxes[i][2]),int(gt_boxes[i][3])),(0,0,255),1)
                cv2.putText(image,str(j),(int(gt_boxes[i][0]),int(gt_boxes[i][1])),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                #cv2.putText()
                #cv2.imwrite(cfg.ROOT_DIR+"/"+file_name+".jpg",image_)
                #print image_
                
                # print 'file_name-------------------------',file_name,cfg.ROOT_DIR+"/test/roi/"+file_name + ".jpg"
        file_name = string.join(random.sample(['a','b','c','d','e','f','g','h','i','j','k','l','m','n'], 6)).replace(" ","")
        cv2.imwrite(cfg.ROOT_DIR+"/test/roi/"+file_name + ".jpg", image)    

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))



        top[0].reshape(*rois_layers[0].shape)
        top[0].data[...] = rois_layers[0]

        top[1].reshape(*rois_layers[1].shape)
        top[1].data[...] = rois_layers[1]

        top[2].reshape(*rois_layers[2].shape)
        top[2].data[...] = rois_layers[2]

        top[3].reshape(*rois_layers[3].shape)
        top[3].data[...] = rois_layers[3]

        # classification labels
        top[4].reshape(*labels.shape)
        top[4].data[...] = labels

        # print 'rois_layers[0].shape',rois_layers[0].shape
        # print 'rois_layers[1].shape',rois_layers[1].shape
        # print 'rois_layers[2].shape',rois_layers[2].shape
        # print 'rois_layers[3].shape',rois_layers[3].shape

        # bbox_targets
        top[5].reshape(*bbox_targets.shape)
        top[5].data[...] = bbox_targets

        # bbox_inside_weights
        top[6].reshape(*bbox_inside_weights.shape)
        top[6].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[7].reshape(*bbox_inside_weights.shape)
        top[7].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        #mask_targets
        top[8].reshape(*mask_targets.shape)
        top[8].data[...] = mask_targets

        #print kps_labels.shape

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

def _find_index_roi_if_not(index,radius,layer_indexs):
    print index,radius
    if index >= 0 and index <= 5:
        left = index-radius
        right = index + radius
        if left >= 0:   
            # indexs = np.where(layer_indexs==left+2)
            print layer_indexs
            index_ = (layer_indexs == (left + 2))
            print index_
            if len(np.where(index_==True)) > 0:
                return index_,True
        elif right <= 3:
            # indexs = np.where(layer_indexs==right+2)
            print layer_indexs
            index_ = (layer_indexs == (right + 2))
            print index_
            if len(np.where(index_ == True)) > 0:
                return indexs_,True
        else:
            return -1,False
        
def _sample_rois(all_rois, gt_boxes,gt_masks,fg_rois_per_image, rois_per_image, num_classes,image_data):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    #print "print all_rois.shape(0) is ", all_rois.shape[0]
    #print all_rois.shape(0)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    gt_boxes_ = gt_boxes[gt_assignment, :4]

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
    print 'fg_inds size',len(fg_inds)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    #print labels.shape
    labels = labels[keep_inds]
    #print labels.shape
    # Clamp labels for the background RoIs to 0
    original_labels = labels
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    #mask_targets = all_mask_targets[keep_inds,:,:,:]
    bboxes = gt_boxes_[keep_inds, :4]
    keep_assingment_inds = gt_assignment[keep_inds]
    mask_targets = np.zeros((len(labels),num_classes-1,cfg.MASK_SIZE,cfg.MASK_SIZE), dtype=np.float32)
    mask_targets[:,:,:,:] = -1

    for i in np.arange(len(labels)):
        if int(labels[i]) == 0:
            mask_targets[i, :, :, :] = -1
        elif int(labels[i]) == int(original_labels[i]):
            x1 = rois[i, 1]
            y1 = rois[i, 2]
            x2 = rois[i, 3]
            y2 = rois[i, 4]
            j = int(labels[i])

            gt_mask = gt_masks[keep_assingment_inds[i],:,:]
            bbox = bboxes[i, :4]

            mask_roi =gt_mask[np.round(y1).astype(np.int):np.round(y2).astype(np.int),np.round(x1).astype(np.int):np.round(x2).astype(np.int)]
            if (int(bbox[2]) - int(bbox[0])) <= 0 or (int(bbox[3]) - int(bbox[1])) <= 0 or bbox[2] < 0 or bbox[
                0] < 0 or bbox[1] < 0 or bbox[3] < 0:
                # resized_mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_CUBIC)
                mask_targets[i, j-1, :, :] = -1
                continue

            if (np.round(y2 - y1).astype(np.int) <= 0) or (np.round(x2 - x1).astype(np.int)) <= 0:
                mask_targets[i, j-1, :, :] = -1
                continue

            if mask_roi.shape[0] <=0 or mask_roi.shape[1] <=0:
                mask_targets[i, j-1, :, :] = -1
                continue

            resized_mask = cv2.resize(mask_roi, (cfg.MASK_SIZE,cfg.MASK_SIZE), interpolation=cv2.INTER_CUBIC)
            resized_mask = np.round(resized_mask)
            mask_targets[i, j-1, :, :] = resized_mask
        else:
            mask_targets[i, j-1, :, :] = -1


    w = (rois[:,3]-rois[:,1])
    h = (rois[:,4]-rois[:,2])
    s = w * h
    s[s<=0]=1e-6
    # layer_index = np.floor(k0+np.log2(np.sqrt(s)/224))
    image_area = image_data.shape[0]*image_data.shape[1]
    alpha = np.sqrt(h * w) / (224.0 / np.sqrt(image_area))
    layer_index_ = np.log(alpha)/np.log(2.0)
    # print np.max(np.hstack((2,4+np.round(layer_index).astype(np.int32)))),'layer_index'
    layer_index=[]
    for i in layer_index_:
        layer_index.append(np.min([5,np.max([2,4+np.round(i).astype(np.int32)])]))
    # print layer_index,'layer_index'
    # roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
    # roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))

    layer_index[layer_index<2]=2
    layer_index[layer_index>5]=5
    layer_indexs = np.array(layer_index)

    # print layer_indexs
    # print rois
    rois_layers=[]
        # mask_targets_=[]
        # num_rois = 0
        # labels_=[]
        # rois_index
        # print gt_masks
        
    gt_index = np.array(range(len(labels)))
    permuted_roi_index_by_layer_ = []
    permuted_mask_targets= np.zeros((len(labels),num_classes-1,cfg.MASK_SIZE,cfg.MASK_SIZE), dtype=np.float32)
    permuted_lables= np.zeros((len(labels)), dtype=np.int)

    # print gt_index

    for i in xrange(4):
        index = (layer_indexs == (i + 2))
        # print i,'-------------------i---------------------'
        # print index
        # print layer_indexs.tolist().index(i+2)
        # print 'index---------------i--------------',index,i
        if np.any(index) == False:
            rois_layers.append(np.array([]))
            permuted_roi_index_by_layer_.append(np.array([]))
            # rois_layers
            # for j in xrange(1,4):
            #     print j,'---------------j-----------------'
            #     index_,found = _find_index_roi_if_not(i,j,layer_indexs)
            #     print index_,found,i,j
            #     if found is True:
            #         rois_layers.append(rois[index_,:][0].reshape(1,5))
            #         mask_targets_.append(mask_targets[index_,:,:,:][0].reshape(1,13,28,28))
            #         labels_.append(labels[index_][0].reshape(1))
            #         num_rois +=1
            #         break
        else:
            rois_layers.append(rois[index,:])
            permuted_roi_index_by_layer_.append(gt_index[index])
            # print gt_index[index]
            # mask_targets_.append(mask_targets[index,:,:,:])
            # labels_.append(labels[index])
            # num_rois += len(index)
    #print rois_layers
    # concat_mask_targets_ = np.zeros(num_rois,num_classes-1,cfg.MASK_SIZE,cfg.MASK_SIZE), dtype=np.float32)
    # for i in xrange(4):
    #     concat_mask_targets_[]
    # for i in xrange(len(mask_targets_)):
    #     print mask_targets_[i].shape
    # concat_mask_targets_ = np.concatenate((mask_targets_), axis=0)
    # print labels_
    # concat_labels_ = np.concatenate((labels_),axis =0)

    # print concat_mask_targets_.shape,'labels shape',concat_labels_.shape
    # print 'num_rois',num_rois
    # sampled rois
    # for i in xrange(4):
    #     print 'rois_layers',rois_layers[i].shape
    # for i in xrange(4):
    #     if len(rois_layers[i]) == 0:
    #         if i < 3:
    #             if rois_layers[i+1].shape[0] > 1:
    #                 rois_layers[i]=rois_layers[i+1][0,:].reshape(1,5)
    #                 rois_layers[i+1]=rois_layers[i+1][1:,:]
    #             else:
    #                 rois_layers[i]=rois_layers[i-1][0,:].reshape(1,5)
    #                 rois_layers[i-1]=rois_layers[i-1][1:,:]
    #         else:
    #             rois_layers[i]=rois_layers[i-1][0,:].reshape(1,5)
    #             rois_layers[i-1]=rois_layers[i-1][1:,:]

    # for i in xrange(4):
    #     print 'rois_layers[]',i,'shape',rois_layers[i].shape

    for i in xrange(4):
        if len(rois_layers[i]) == 0:
            index = i
            if index-1 >=0 and rois_layers[index-1].shape[0] > 1:
                len_rois_layers = rois_layers[index-1].shape[0]
                rois_layers[i]=rois_layers[index-1][len_rois_layers-1,:].reshape(1,5)
                rois_layers[index-1]=rois_layers[index-1][0:len_rois_layers-1,:]
            elif index+1 < 4 and rois_layers[index+1].shape[0] > 1:
                rois_layers[i]=rois_layers[index+1][0,:].reshape(1,5)
                rois_layers[index+1]=rois_layers[index+1][1:,:]
            elif index-2 >=0 and rois_layers[index-2].shape[0] > 1:
                len_rois_layers = rois_layers[index-2].shape[0]
                # print len_rois_layers,'eeeeeeeeeeeee',index
                rois_layers[i]=rois_layers[index-1][0,:].reshape(1,5)
                rois_layers[index-1]=rois_layers[index-2][len_rois_layers-1,:].reshape(1,5)
                # rois_layers[i]=rois_layers[index-2][0,:].reshape(1,5)
                rois_layers[index-2]=rois_layers[index-2][0:len_rois_layers-1,:]
            elif index+2 < 4 and rois_layers[index+2].shape[0] > 1 :
                # print rois_layers[index+1]
                # print rois_layers[index+1][0,:]
                # print rois_layers[index+2][0,:]
                # print rois_layers[index+2]
                if rois_layers[index+1].shape[0] == 0:
                    rois_layers[i+1]=rois_layers[index+2][1,:].reshape(1,5)
                    rois_layers[i]=rois_layers[index+2][0,:].reshape(1,5)
                    rois_layers[index+2]=rois_layers[index+2][2:,:]
                else:
                    rois_layers[i]=rois_layers[index+1][0,:].reshape(1,5)
                    rois_layers[i+1]=rois_layers[index+2][0,:].reshape(1,5)
                    rois_layers[index+2]=rois_layers[index+2][1:,:]
            elif index-3 >=0 and rois_layers[index-3].shape[0] > 1:
                len_rois_layers = rois_layers[index-3].shape[0]
                # print len_rois_layers,'ddddddddddddd',index
                rois_layers[i]=rois_layers[index-1][0,:].reshape(1,5)
                rois_layers[index-1]=rois_layers[index-2][0,:].reshape(1,5)
                rois_layers[index-2]=rois_layers[index-3][len_rois_layers-1,:]
                # rois_layers[i]=rois_layers[index-2][0,:].reshape(1,5)
                rois_layers[index-3]=rois_layers[index-3][0:len_rois_layers-1,:]
            elif index+3 < 4 and rois_layers[index+3].shape[0] > 1 :
                len_rois_layers = rois_layers[index+3].shape[0]
                rois_layers[i]=rois_layers[index+1][0,:].reshape(1,5)
                rois_layers[index+1]=rois_layers[index+2][0,:].reshape(1,5)
                rois_layers[index+2]=rois_layers[index+3][0,:].reshape(1,5)
                # rois_layers[i]=rois_layers[index-2][0,:].reshape(1,5)
                rois_layers[index+3]=rois_layers[index+3][1:,:]
            


        # print 'rois_layers',rois_layers[i].shape
    
    # for i in xrange(4):
    #     print 'rois_layers[]',i,'shape',rois_layers[i].shape
    # permuted_roi_index_by_layer_ = []
    rois_gt_boxes = gt_boxes[gt_assignment[keep_inds], :4]
    permuted_mask_targets= np.zeros((len(labels),num_classes-1,cfg.MASK_SIZE,cfg.MASK_SIZE), dtype=np.float32)
    permuted_lables= np.zeros((len(labels)), dtype=np.int)
    permuted_gt_boxes = np.zeros((len(labels),4), dtype=np.float32)
    permuted_rois = np.zeros((len(labels),5), dtype=np.float32)
    count =0
    for i in xrange(4):
        permuted_index_i = permuted_roi_index_by_layer_[i]
        count_layer_i = len(permuted_index_i)
        if count_layer_i != 0 :
            permuted_lables[count:count_layer_i+count] = labels[permuted_index_i]
            permuted_mask_targets[count:count_layer_i+count,:,:,:]= mask_targets[permuted_index_i,:,:,:]
            permuted_gt_boxes[count:count_layer_i+count] = rois_gt_boxes[permuted_index_i,:4]
            permuted_rois[count:count_layer_i+count]=rois[permuted_index_i,:5]
            count += count_layer_i

    bbox_target_data = _compute_targets(
        permuted_rois[:, 1:5], permuted_gt_boxes, permuted_lables)
    # bbox_target_data = _compute_targets(
    #     rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    # return labels, rois, bbox_targets, bbox_inside_weights, mask_targets,layer_indexs
    return permuted_lables, permuted_rois, permuted_gt_boxes,bbox_targets, bbox_inside_weights, permuted_mask_targets,rois_layers
