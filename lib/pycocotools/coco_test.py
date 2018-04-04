#
#import _init_paths
import sys
sys.path.append("..")
from fast_rcnn.config import cfg
from coco import COCO
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

class COCOTest():
    def __init__(self, image_set, year):
        #imdb.__init__(self, 'coco_' + year + '_' + image_set)
        # COCO specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'crowd_thresh' : 0.7,
                       'min_size' : 2}
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'coco')
        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        #print self._COCO.anns[185487]
        # anns = [self._COCO.anns[185487]]
        #self._COCO.showAnns(anns)
        #image_ids = self._COCO.getImgIds()
        #print image_ids
        self.test()

    def _get_ann_file(self):
        # prefix = 'instances' if self._image_set.find('test') == -1 \
        #                      else 'image_info'
        # return osp.join(self._data_path, 'annotations',
        #                 prefix + '_' + self._image_set + self._year + '.json')
        return osp.join(self._data_path,'annotations','person_keypoints_train2014.json')


    def get_img_file(self,im_ann):
        return osp.join(self._data_path, 'train2014',im_ann['file_name'])

    def test(self):
        image_ids = self._COCO.getImgIds()
        # print image_ids,'\n,len:',len(image_ids)
        for i in xrange(len(image_ids)):
            im_ann = self._COCO.loadImgs(image_ids[i])[0]
            print '\n:',i
            width = im_ann['width']
            height = im_ann['height']
            # print im_ann
            # print self.get_img_file(im_ann)

            annIds = self._COCO.getAnnIds(imgIds=image_ids[i], iscrowd=None)
            objs = self._COCO.loadAnns(annIds)
            # print annIds,objs
            im = cv2.imread(self.get_img_file(im_ann))
            # Sanitize bboxes -- some are invalid
            valid_objs = []
            im = im[:, :, (2, 1, 0)]
            im[:,:,:]=(0,0,0)
            #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            fig, ax = plt.subplots(figsize=(12, 12))
            for obj in objs:
                print obj
                mask = self._COCO.annToMask(obj)
                #im[mask==0,:]=(0,0,0);
                im[mask == 1, :] = (255, 255, 255);
                # for i in range(width):
                #     for j in range(height):
                #         if(mask[i][j] == 0)

                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
                # print mask.shape,x1,y1,x2,y2,width,height
                # print 'mask.shape[0]:mask_shape[1]',mask.shape[0],mask.shape[1]
                start_h = np.round(np.max((1,y1))).astype(np.int)
                end_h = np.round(np.min((height,y2))).astype(np.int)
                start_w = np.round(np.max((1,x1))).astype(np.int)
                end_w = np.round(np.min((width,x2))).astype(np.int)
                cropped_mask = mask[start_h:end_h, start_w:end_w]
                # print cropped_mask.shape
                # resize_mask = cv2.resize(cropped_mask, (28, 28), interpolation=cv2.INTER_NEAREST)
                # print resize_mask.shape,resize_mask
                if 'keypoints' in obj:
                    print 'category_id',obj['category_id']
                    print '\nkeypoints',obj['keypoints']
                    print '\nlens',obj['num_keypoints']

        # cropped_mask = mask[np.max(1,int(y1)):np.min(height,int(y2)),np.max(1, int(x1)):np.min(width,int(x2))]
        #     print mask
        #     if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        #         obj['clean_bbox'] = [x1, y1, x2, y2]
        #         valid_objs.append(obj)
        #         ax.add_patch(
        #             plt.Rectangle((x1, y1),
        #                           x2-x1,
        #                           y2-y1, fill=False,
        #                           edgecolor='red', linewidth=3.5)
        #         )
        #         ax.text(x1, y1 - 2,
        #                 '{:d} {:.3f}'.format(obj['category_id'], 1.0),
        #                 bbox=dict(facecolor='blue', alpha=0.5),
        #                 fontsize=14, color='white')
        #
        # ax.imshow(im, aspect='equal')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.draw()
        # plt.show()
        # objs = valid_objs
        # num_objs = len(objs)




if __name__ == '__main__':
    coco = COCOTest('train','2014')
    print 'test'
