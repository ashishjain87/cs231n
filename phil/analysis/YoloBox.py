from typing import List, Tuple, Optional
from matplotlib.pyplot import box
from collections import defaultdict

import numpy as np


class YoloBox:
    category: int
    center_x: float
    center_y: float
    width: float
    height: float

    def __init__(self, category: int, center_x: float, center_y: float, width: float, height: float):
        self.category = category
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

    def size(self):
        return self.width * self.height

    def iou(self, other) -> Tuple[bool, float]:
        """

        :param other: another yolobox
        :return: a bool indicating whether the two boxes overlap and an IoU score if they do overlap
        """
        if type(other) != YoloBox:
            raise ValueError("must pass in another YoloBox")

        return overlap(self, other)

    def intersects(self, other) -> bool:
        if type(other) != YoloBox:
            raise ValueError("must pass in another YoloBox")
        return self.iou(other)[0]


def overlap(a: YoloBox, b: YoloBox) -> Tuple[bool, float]:
    """

    :param a: a YoloLabel
    :param b: a different YoloLabel
    :return: a bool indicating whether the two labels overlap and an IoU score if they do overlap
    """
    x_intersection = intersection((a.center_x - a.width/2, a.center_x + a.width/2),
                                  (b.center_x - b.width/2, b.center_x + b.width/2))
    y_intersection = intersection((a.center_y - a.height / 2, a.center_y + a.height / 2),
                                  (b.center_y - b.height / 2, b.center_y + b.height / 2))
    if x_intersection is None or y_intersection is None:
        return False, 0.

    intersect_area = (x_intersection[1] - x_intersection[0])*(y_intersection[1] - y_intersection[0])
    a_area = a.width * a.height
    b_area = b.width * b.height
    iou = intersect_area/(a_area + b_area - intersect_area)

    return True, iou


def intersection(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    a_s, a_e = a
    b_s, b_e = b
    if a_s <= b_e and b_s <= a_e:
        overlap = max(a_s, b_s), min(a_e, b_e)
        if overlap[0] == overlap[1]:
            return None
        else:
            return overlap
    return None


def get_boxes_from_nparray(label: np.ndarray) -> List[YoloBox]:
    """

    :param label:
    :return:
    """
    if len(label.shape) == 1:
        label = label.reshape(1, -1)
    elif label.shape[1] == 0:
        return []

    boxes = []
    for i in range(label.shape[0]):
        vector = label[i, :]
        category = vector[0]
        center_x = vector[1]
        center_y = vector[2]
        width = vector[3]
        height = vector[4]
        boxes.append(YoloBox(category, center_x, center_y, width, height))

    return boxes

def match_boxes_trivial(box_set_1: List[YoloBox], box_set_2: List[YoloBox]) -> List[Tuple[YoloBox, Optional[YoloBox]]]:
    """
    same interface as match_boxes, but sets are of same length and correspond to the other. can just return zip(set1, set2) (test this though, please).

    :return matching:
    """
    assert len(box_set_1) == len(box_set_2), "Modal and Amodal boxes have different number of labels."

    matching = list(zip(box_set_1, box_set_2))
    return matching


def match_boxes(box_set_1: List[YoloBox], box_set_2: List[YoloBox]) -> List[Tuple[YoloBox, Optional[YoloBox]]]:
    """
    attempts to find the best box in box_set_2 for each box in box_set_1. duplicate assignments are not allowed!

    Note: don't actually have to run this for amodal/modal pairs, structure of label file already gives us the pairing

    :param box_set_1:
    :param box_set_2:
    :return:
    """

    matching = []

    box_pairs_list = []
    for i, box1 in enumerate(box_set_1):
        for j, box2 in enumerate(box_set_2):
            overlap, iou = box1.iou(box2)
            if overlap:
                box_pairs_list.append((box1, box2, iou, i, j))

    box_pairs_list = sorted(box_pairs_list, key=lambda x: x[2], reverse=True)

    mapping = defaultdict(lambda: None)
    used_i = set()
    used_j = set()
    for box1, box2, iou, i, j in box_pairs_list:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        mapping[i] = j

    for i, box1 in enumerate(box_set_1):
        j = mapping[i]
        if j is None:
            matching.append((box1, None))
        else:
            matching.append((box1, box_set_2[j]))

    return matching
