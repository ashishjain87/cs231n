import numpy as np

from phil.analysis import BoxplotHistogramCollector


class ComparativeExampleMetric:
    """
    An abstract class indicating a pattern for using the boxplot collector
    """

    collector: BoxplotHistogramCollector

    def __init__(self, collector: BoxplotHistogramCollector):
        self.collector = collector

    def process_image(self, amodal_label: np.ndarray, modal_label: np.ndarray,
                      prediction_label_1: np.ndarray, prediction_label_2: np.ndarray):
        raise NotImplementedError("")
