from abc import ABC, abstractmethod
from PIL import Image 

class Rotator(ABC):
    @abstractmethod
    def rotate(
        self,
        foregroundImage: Image
    ) -> Image:
        pass

