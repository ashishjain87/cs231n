import argparse
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import random
from phil import analyse


def iou_test_no_overlap():
    test_mask1 = np.zeros((10,10))
    test_mask1[2:5,2:5] = 1

    test_mask2 = np.zeros((10,10))
    test_mask2[7:9,7:9] = 1
    IoU = analyse.compute_segmentaion_IoU(test_mask1, test_mask2)
    
    assert IoU == 0.0
    print(f"test_mask1:\n{test_mask1}")
    print(f"test_mask2:\n{test_mask2}")
    print(IoU)

def iou_test_one_column_overlap():
    test_mask1 = np.zeros((10,10))
    test_mask1[2:5,2:5] = 1

    test_mask2 = np.zeros((10,10))
    test_mask2[2:5,4:6] = 1
    IoU = analyse.compute_segmentaion_IoU(test_mask1, test_mask2)
    
    assert IoU == 0.25
    print(f"test_mask1:\n{test_mask1}")
    print(f"test_mask2:\n{test_mask2}")
    print(f"overlap:\n{test_mask1 + test_mask2}")
    print(IoU)

def iou_test_full_overlap():
    test_mask1 = np.zeros((10,10))
    test_mask1[2:5,2:5] = 1

    test_mask2 = np.zeros((10,10))
    test_mask2[2:5,2:5] = 1
    IoU = analyse.compute_segmentaion_IoU(test_mask1, test_mask2)
    
    assert IoU == 1.
    print(f"test_mask1:\n{test_mask1}")
    print(f"test_mask2:\n{test_mask2}")
    print(f"overlap:\n{test_mask1 + test_mask2}")
    print(IoU)

def iou_test_two_objects_per_mask():
    test_mask1 = np.zeros((10,10))
    test_mask1[0:3,0:3] = 1
    test_mask1[6:10,6:10] = 1

    test_mask2 = np.zeros((10,10))
    test_mask2[0:3,0:3] = 1
    test_mask2[6:10,0:3] = 1
    IoU = analyse.compute_segmentaion_IoU(test_mask1, test_mask2)
    
    assert IoU == 9./37
    print(f"test_mask1:\n{test_mask1}")
    print(f"test_mask2:\n{test_mask2}")
    print(f"overlap:\n{test_mask1 + test_mask2}")
    print(IoU)

def iou_test_two_objects_per_mask_partial_overlap():
    test_mask1 = np.zeros((10,10))
    test_mask1[0:3,0:3] = 1
    test_mask1[6:10,6:10] = 1

    test_mask2 = np.zeros((10,10))
    test_mask2[0:3,2:6] = 1
    test_mask2[6:10,3:7] = 1
    IoU = analyse.compute_segmentaion_IoU(test_mask1, test_mask2)
    
    assert IoU == 7./46
    print(f"test_mask1:\n{test_mask1}")
    print(f"test_mask2:\n{test_mask2}")
    print(f"overlap:\n{test_mask1 + test_mask2}")
    print(IoU)

if __name__ == "__main__":
    # iou_test_no_overlap()
    # iou_test_one_column_overlap()
    # iou_test_full_overlap()
    # iou_test_two_objects_per_mask()
    iou_test_two_objects_per_mask_partial_overlap()