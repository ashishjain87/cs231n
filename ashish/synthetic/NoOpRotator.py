from PIL import Image 
from Rotator import Rotator

class NoOpRotator(Rotator):
    def rotate(
        self,
        foregroundImage: Image
    ) -> Image:
        return foregroundImage

