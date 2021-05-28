import numpy as np
from abc import ABC
from typing import Tuple
from PIL import Image
from Affixer import Affixer

class FixedAffixer(Affixer):
    def decide_where_and_scale(
        self,
        backgroundImage: Image,
        backgroundAnnotations: np.ndarray,
        foregroundImage: Image
    ) -> Tuple[Tuple[int, int], float]:
        return ((100, 100), 1.)

