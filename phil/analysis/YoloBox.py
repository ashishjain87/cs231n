from typing import List, Tuple, Optional

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
    elif label[1] == 0:
        return []

    boxes = []
    for i in range(label.shape[0]):
        vector = label[0, :]
        category = vector[0]
        center_x = vector[1]
        center_y = vector[2]
        width = vector[3]
        height = vector[4]
        boxes.append(YoloBox(category, center_x, center_y, width, height))

    return boxes


