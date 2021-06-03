import numpy as np

from phil.analysis import BoxplotHistogramCollector, Todos
from phil.analysis import YoloBox
from phil.analysis.SingleExampleMetric import SingleExampleMetric


class SeverityVsGoodness(SingleExampleMetric):

    def __init__(self, collector: BoxplotHistogramCollector):
        super().__init__(collector)

    def process_image(self, amodal_label: np.ndarray, modal_label: np.ndarray,
                      prediction_label: np.ndarray):
        amodal_boxes = YoloBox.get_boxes_from_nparray(amodal_label)
        modal_boxes = YoloBox.get_boxes_from_nparray(modal_label)
        pred_boxes = YoloBox.get_boxes_from_nparray(prediction_label)

        gt_box_pairs = YoloBox.match_boxes_trivial(amodal_boxes, modal_boxes)
        pred_box_pairs = YoloBox.match_boxes(amodal_boxes, pred_boxes)

        for gt_pair, pred_pair in zip(gt_box_pairs, pred_box_pairs):
            if pred_pair[1] is None:
                #
                iou = 0
            else:
                intersect, iou = pred_pair[0].iou(pred_pair[1])
            comparative_score = gt_pair[1].size()/gt_pair[0].size()  # this is a measurement of occlusion severity
            self.collector.record_event(iou, comparative_score)
