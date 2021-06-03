from typing import List, Tuple, Optional

from phil.analysis.YoloBox import YoloBox


def match_boxes_trivial(box_set_1: List[YoloBox], box_set_2: List[YoloBox]) -> List[Tuple[YoloBox, Optional[YoloBox]]]:
    """
    same interface as match_boxes, but sets are of same length and correspond to the other. can just return zip(set1, set2) (test this though, please).

    :return:
    """
    raise NotImplementedError()


def match_boxes(box_set_1: List[YoloBox], box_set_2: List[YoloBox]) -> List[Tuple[YoloBox, Optional[YoloBox]]]:
    """
    attempts to find the best box in box_set_2 for each box in box_set_1. duplicate assignments are not allowed!

    Note: don't actually have to run this for amodal/modal pairs, structure of label file already gives us the pairing

    :param box_set_1:
    :param box_set_2:
    :return:
    """
    raise NotImplementedError()
