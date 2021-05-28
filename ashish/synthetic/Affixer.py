import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from PIL import Image 

class Affixer(ABC):
    @abstractmethod
    def decide_where_and_scale(
        self,
        backgroundImage: Image,
        backgroundAnnotations: np.ndarray,
        foregroundImage: Image
    ) -> Tuple[Tuple[int, int], float]:
        pass
