import numpy as np
from imwatermark import WatermarkDecoder
from PIL import Image, ImageDraw
import random

def interpolate_img(org_img, candidate_img, alpha=0.2, mask_image=None):
    return Image.composite(candidate_img, org_img, mask_image.convert("L")) 

def generate_square_mask(image, mask_ratio=0.25):
    # Calculate the mask size based on the mask_ratio
    canvas_size = image.size
    mask_area = canvas_size[0] * canvas_size[1] * mask_ratio
    mask_side_length = int(mask_area**0.5)  # Assuming a square mask

    mask = Image.new('L', canvas_size, 0)  # Create a black image for the canvas

    # Generate random top left corner for the mask
    x = random.randint(0, canvas_size[0] - mask_side_length)
    y = random.randint(0, canvas_size[1] - mask_side_length)

    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + mask_side_length, y + mask_side_length], fill=255)  # Draw a white square
    return mask

# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
MATCH_VALUES = [
    [27, "No watermark detected"],
    [33, "Partial watermark match. Cannot determine with certainty."],
    [
        35,
        (
            "Likely watermarked. In our test 0.02% of real images were "
            'falsely detected as "Likely watermarked"'
        ),
    ],
    [
        49,
        (
            "Very likely watermarked. In our test no real images were "
            'falsely detected as "Very likely watermarked"'
        ),
    ],
]


class GetWatermarkMatch:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(self.watermark)
        self.decoder = WatermarkDecoder("bits", self.num_bits)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Detects the number of matching bits the predefined watermark with one
        or multiple images. Images should be in cv2 format, e.g. h x w x c BGR.

        Args:
            x: ([B], h w, c) in range [0, 255]

        Returns:
           number of matched bits ([B],)
        """
        squeeze = len(x.shape) == 3
        if squeeze:
            x = x[None, ...]

        bs = x.shape[0]
        detected = np.empty((bs, self.num_bits), dtype=bool)
        for k in range(bs):
            detected[k] = self.decoder.decode(x[k], "dwtDct")
        result = np.sum(detected == self.watermark, axis=-1)
        if squeeze:
            return result[0]
        else:
            return result

get_watermark_match = GetWatermarkMatch(WATERMARK_BITS)