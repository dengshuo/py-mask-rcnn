import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict as edict
import numpy as np

__D = edict()

debug = __D

__D.TRAIN = edict()
__D.TRAIN.CURRENT_DATA = np.zeros((1000,600),dtype=np.uint8)
__D.TRAIN.DEBUG=True




