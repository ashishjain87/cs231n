import math
import random
import Annotation
import Occlusions
import SyntheticImage

from PIL import Image 
from Rotator import Rotator

class RandomRotator(Rotator):
    def rotate(
        self,
        foregroundImage: Image
    ) -> Image:
        # Create a new image which will not drop positive pixels simply because they go out of the image frame when the image is rotated. 
        expandedImage = RandomRotator.expand_image_as_per_diagonal(foregroundImage) 

        # Decide degree to rotate
        angle = random.randint(0,360) # [a,b] => 0 has two times the chance of any other value

        # Rotate the image
        rotatedImage = expandedImage.rotate(angle)

        # Crop the image to remove empty space
        croppedImage = RandomRotator.remove_empty_space_around_edges(rotatedImage)

        return croppedImage

    @staticmethod
    def expand_image_as_per_diagonal(image: Image) -> Image:
        # Create blank square image of size diagonal*2
        width, height = image.size
        diagX, diagY= math.ceil(width/2), math.ceil(height/2)
        diagLength = math.sqrt(diagX**2 + diagY**2)
        widthExpanded = heightExpanded = diagLength*2
        blankImage = Image.new('RGBA', (widthExpanded, heightExpanded), (0, 0, 0, 0))
        centerExpanded = (diagLength, diagLength)
        syntheticImage, topLeftPoint, mask = SyntheticImage.create_synthetic_image(blankImage, image, 100, centerExpanded)
        return syntheticImage

    @staticmethod
    def remove_empty_space_around_edges(expandedImage: Image) -> Image:
        mask = Occlusions.create_grayscale_image_mask(expandedImage, 100)
        width, height = expandedImage.size
        (topLeftPointMaskWithinForegroundImageX, topLeftPointMaskWithinForegroundImageY, maskWidthWithinForegroundImage, maskHeightWithinForegroundImage) = Annotation.get_bounding_box(np.array(mask))
        croppedImage = expandedImage.crop((
            max(0, topLeftPointMaskWithinForegroundImageX-1),
            max(0, topLeftPointMaskWithinForegroundImageY-1),
            min(topLeftPointMaskWithinForegroundImageX+maskWidthWithinForegroundImage+1, width),
            min(topLeftPointMaskWithinForegroundImageY+maskHeightWithinForegroundImage+1, height)))
        return croppedImage

