from PIL import Image, ImageDraw, ImageFont
import string
import numpy as np
import cv2
from helpers import calcCentroids
from performance import GlobalTimer

class TemplateGenerator:
    def __init__(self, font_file=None, font_size=8, patch_w=8, patch_h=8, save=True, char_set=' .*:o&8?'):
        self.FONT_FILE = font_file
        self.FONT_SIZE = font_size
        self.PATCH_W, self.PATCH_H = patch_w, patch_h
        self.SAVE=save
        self.char_set = char_set

    @GlobalTimer.time
    def generate(self):

        FONT_FILE = self.FONT_FILE
        FONT_SIZE = self.FONT_SIZE
        PATCH_W, PATCH_H = self.PATCH_W, self.PATCH_H
        CHARS = "".join(set('|-/\\') | self.char_set) # must have at least the edge characters
        TEMP_CANVAS_SIZE = 100
        template_library = {}
        centroids = {}


        try:
            font = ImageFont.truetype("templates/font.ttf", FONT_SIZE)
            print(f"Generating font with size {FONT_SIZE}")
        except IOError:
            print(f"Could not load font '{FONT_FILE}'. Using default.")
            font = ImageFont.load_default()

        for char in CHARS:

            # Draw character to a big temporary canvas to prevent clipping
            temp_img = Image.new('L', (TEMP_CANVAS_SIZE, TEMP_CANVAS_SIZE), 0)
            temp_draw = ImageDraw.Draw(temp_img)

            temp_draw.text(
                (TEMP_CANVAS_SIZE / 2, TEMP_CANVAS_SIZE / 2),
                char,
                font=font,
                fill=255,
                anchor="mm" # draw in middle
                )

            # Calculate the visual bounding box for this character
            bounding_box = temp_img.getbbox()

            # Crop template image to the bounding box (or don't if image is empty)
            if bounding_box is None:
                cropped_char = Image.new('L', (PATCH_W, PATCH_H), 0)
            else:
                cropped_char = temp_img.crop(bounding_box)

            # Paste template onto a new image with the intended patch size
            final_template = Image.new('L', (PATCH_W, PATCH_H), 0)
            paste_x = (PATCH_W - cropped_char.width) // 2
            paste_y = (PATCH_H - cropped_char.height) // 2
            final_template.paste(cropped_char, (paste_x, paste_y))

            # Normalize template to 0.0 - 1.0 to match input image range
            template = np.array(final_template).astype(np.float32) / 255.0

            centroid = calcCentroids(template)

            centroids[char] = centroid
            template_library[char] = template

            if self.SAVE:
                final_template.save(f"./templates/template_{ord(char)}.png")


        print(f"Template generation done.")

        return template_library, centroids
