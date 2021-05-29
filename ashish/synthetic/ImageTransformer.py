from abc import ABC, abstractmethod
from PIL import Image 

class ImageTransformer(ABC):
    @abstractmethod
    def transform(
        self,
        foregroundImage: Image
    ) -> Image:
        pass

