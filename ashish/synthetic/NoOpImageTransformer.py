from PIL import Image 
from ImageTransformer import ImageTransformer 

class NoOpImageTransformer(ImageTransformer):
    def transform(
        self,
        foregroundImage: Image
    ) -> Image:
        return foregroundImage

