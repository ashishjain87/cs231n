import numpy as np
from typing import Tuple
from PIL import Image
import random

from Affixer import Affixer
from OriginalAffixer import OriginalAffixer

DEFAULT_LEAST_OCCLUSION_SEVERITY = 0.25
DEFAULT_GREATEST_OCCLUSION_SEVERITY = 0.75

DEFAULT_MAX_SCALE_MULTIPLIER = 1.8  # max scale is this much times min scale


class SideAffixer(Affixer):
    """
    Affixes foreground images to background images such that at least one object's modal box differs because of the
    new occlusion.
    """
    least_occlusion_severity: float
    greatest_occlusion_severity: float
    max_scale_multiplier: float

    def __init__(self, least_occlusion_severity: float = DEFAULT_LEAST_OCCLUSION_SEVERITY,
                 greatest_occlusion_severity: float = DEFAULT_GREATEST_OCCLUSION_SEVERITY,
                 max_scale_multiplier: float = DEFAULT_MAX_SCALE_MULTIPLIER):
        super().__init__()
        self.least_occlusion_severity = least_occlusion_severity
        self.greatest_occlusion_severity = greatest_occlusion_severity
        self.max_scale_multiplier = max_scale_multiplier
        self.assert_constructor_params()

    def assert_constructor_params(self):
        assert(0 <= self.least_occlusion_severity <= 1)
        assert(0 <= self.greatest_occlusion_severity <= 1)
        assert(self.least_occlusion_severity <= self.greatest_occlusion_severity)
        assert(self.max_scale_multiplier >= 1)


    def decide_where_and_scale(
        self,
        backgroundImage: Image,
        backgroundAnnotations: np.ndarray,
        foregroundImage: Image
    ) -> Tuple[Tuple[int, int], float]:
        occlusion_severity = random.uniform(self.least_occlusion_severity, self.greatest_occlusion_severity)

        object_idx = OriginalAffixer.randomly_choose_object_of_interest(len(backgroundAnnotations))
        top_left, bottom_right = OriginalAffixer.get_top_left_bottom_right_coordinates(backgroundAnnotations, object_idx, backgroundImage)

        foreground_width, foreground_height = foregroundImage.size

        span_vertical = random.randint(0, 2) == 0

        if span_vertical:
            # first, determine horizontal position of occluder edge
            obj_width = bottom_right[0] - top_left[0]
            occlusion_distance = int(obj_width * occlusion_severity)

            # equally likely to come in from the right or come in from the left
            slide_neg = random.randint(0, 2) == 0
            if slide_neg:
                occluder_edge = occlusion_distance + top_left[0]
            else:
                occluder_edge = bottom_right[0] - occlusion_distance

            min_occluder_scale = max(occlusion_distance/foreground_width, (bottom_right[1] - top_left[1])/foreground_height)
            max_occluder_scale = self.max_scale_multiplier * min_occluder_scale

            occluder_scale = random.uniform(min_occluder_scale, max_occluder_scale)

            if slide_neg:
                occluder_left = occluder_edge - occluder_scale * foreground_width
            else:
                occluder_left = occluder_edge

            max_occluder_top = top_left[1]
            min_occluder_top = top_left[1] + (bottom_right[1] - top_left[1]) - int(occluder_scale * foreground_height)

            occluder_top = random.randint(min_occluder_top, max_occluder_top + 1)

            center_x = int(occluder_left + occluder_scale*foreground_width/2)
            center_y = int(occluder_top + occluder_scale*foreground_height/2)

            return ((center_x, center_y), occluder_scale)

        else:
            # first, determine vertical position of occluder edge
            obj_height = bottom_right[1] - top_left[1]
            occlusion_distance = int(obj_height * occlusion_severity)

            # equally likely to come in from the right or come in from the left
            slide_neg = random.randint(0, 2) == 0
            if slide_neg:
                occluder_edge = occlusion_distance + top_left[1]
            else:
                occluder_edge = bottom_right[1] - occlusion_distance

            min_occluder_scale = max(occlusion_distance / foreground_height,
                                     (bottom_right[0] - top_left[0]) / foreground_width)
            max_occluder_scale = self.max_scale_multiplier * min_occluder_scale

            occluder_scale = random.uniform(min_occluder_scale, max_occluder_scale)

            if slide_neg:
                occluder_top = occluder_edge - occluder_scale * foreground_height
            else:
                occluder_top = occluder_edge

            max_occluder_left = top_left[0]
            min_occluder_left = top_left[0] + (bottom_right[0] - top_left[0]) - int(occluder_scale * foreground_width)

            occluder_left = random.randint(min_occluder_left, max_occluder_left + 1)

            center_x = int(occluder_left + occluder_scale * foreground_width / 2)
            center_y = int(occluder_top + occluder_scale * foreground_height / 2)

            return ((center_x, center_y), occluder_scale)
