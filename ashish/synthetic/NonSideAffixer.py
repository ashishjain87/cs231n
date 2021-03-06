import numpy as np
from typing import Tuple
from PIL import Image
import random

from Affixer import Affixer
from OriginalAffixer import OriginalAffixer

DEFAULT_MIN_NON_OVERLAP = 0.15
DEFAULT_MIN_OVERLAP = 0.30

class NonSideAffixer(Affixer):
    """
    Affixes foreground images to background images such that no modal boxes differ because of the
    new occlusion.
    """
    min_percentage_of_background_height: float
    max_percentage_of_background_height: float
    min_non_overlap: float
    min_overlap: float

    def  __init__(self, min_percentage_of_background_height: float = 10.0, max_percentage_of_background_height: float = 40.0,
                  min_non_overlap: float = DEFAULT_MIN_NON_OVERLAP, min_overlap: float = DEFAULT_MIN_OVERLAP) -> None:
        super().__init__()
        assert min_percentage_of_background_height > 0 and min_percentage_of_background_height < 100, "min_percentage_of_background_height cannot be negative"
        assert max_percentage_of_background_height > 0 and max_percentage_of_background_height < 100, "max_percentage_of_background_height cannot be negative"
        assert min_percentage_of_background_height <= max_percentage_of_background_height, "min_percentage_of_background_height cannot be greater than max_percentage_of_background_height"
        assert (0 < min_non_overlap < 1)
        assert (0 < min_overlap < 1)
        assert (0 < (min_overlap + min_non_overlap) < 1), "sum of min_overlap and min_non_overlap MUST be < 1!"
        self.min_percentage_of_background_height = min_percentage_of_background_height
        self.max_percentage_of_background_height = max_percentage_of_background_height
        self.min_non_overlap = min_non_overlap  # increase this to prevent certain high occlusion severities
        self.min_overlap = min_overlap  # increase this to prevent certain low occlusion severities

    def decide_where_and_scale(
        self,
        backgroundImage: Image,
        backgroundAnnotations: np.ndarray,
        foregroundImage: Image
    ) -> Tuple[Tuple[int, int], float]:
        object_idx = OriginalAffixer.randomly_choose_object_of_interest(len(backgroundAnnotations))
        top_left, bottom_right = OriginalAffixer.get_top_left_bottom_right_coordinates(backgroundAnnotations, object_idx, backgroundImage)

        obj_width = (bottom_right[0] - top_left[0])
        obj_height = (bottom_right[1] - top_left[1])
        background_width, background_height = backgroundImage.size
        foreground_width, foreground_height = foregroundImage.size


        min_horiz_non_overlap = max(1, self.min_non_overlap * obj_width)
        min_vert_non_overlap = max(1, self.min_non_overlap * obj_height)

        min_horiz_overlap = max(1, self.min_overlap * obj_width)
        min_vert_overlap = max(1, self.min_overlap * obj_width)

        jittered_top_left, jittered_bot_right = top_left, bottom_right
        jitter_type = random.randint(0, 4)
        if jitter_type == 0:
            # collapse image to the left by nonoverlap and down by one nonoverlap
            jittered_top_left = (top_left[0] + min_horiz_non_overlap, top_left[1] + min_vert_non_overlap)
            jittered_bot_right = (bottom_right[0] - min_horiz_overlap, bottom_right[1] - min_vert_overlap)
        elif jitter_type == 1:
            # collapse image to the right by nonoverlap and up by nonoverlap
            jittered_bot_right = (bottom_right[0] - min_horiz_non_overlap, bottom_right[1] - min_vert_non_overlap)
            jittered_top_left = (top_left[0] + min_horiz_overlap, top_left[1] + min_vert_overlap)
        elif jitter_type == 2:
            # collapse image to the left by nonoverlap and up by one nonoverlap
            jittered_top_left = (top_left[0] + min_horiz_non_overlap, top_left[1] - min_vert_overlap)
            jittered_bot_right = (bottom_right[0] - min_horiz_overlap, bottom_right[1] + min_vert_non_overlap)
        else:
            # collapse image to the right by nonoverlap and down by one nonoverlap
            jittered_bot_right = (bottom_right[0] + min_horiz_non_overlap, bottom_right[1] - min_vert_overlap)
            jittered_top_left = (top_left[0] + min_horiz_overlap, top_left[1] + min_vert_non_overlap)

        # take new point
        sampled_x, sampled_y = OriginalAffixer.randomly_sample_point_within_rectangle(jittered_top_left, jittered_bot_right)

        min_scale = self.min_percentage_of_background_height * background_height * 0.01 / foreground_height
        max_scale = self.max_percentage_of_background_height * background_height * 0.01 / foreground_height

        scale = random.uniform(min_scale, max_scale)

        if jitter_type == 0:
            # sampled point is top left
            centerx = int(sampled_x + scale*foreground_width/2)
            centery = int(sampled_y + scale*foreground_height/2)
        elif jitter_type == 1:
            # sampled point is bottom right
            centerx = int(sampled_x - scale*foreground_width/2)
            centery = int(sampled_y - scale*foreground_height/2)
        elif jitter_type == 2:
            # sampled point is bottom left
            centerx = int(sampled_x + scale * foreground_width / 2)
            centery = int(sampled_y - scale * foreground_height / 2)
        else:
            # sampled point is top right
            centerx = int(sampled_x - scale * foreground_width / 2)
            centery = int(sampled_y + scale * foreground_height / 2)

        return ((centerx, centery), scale)
