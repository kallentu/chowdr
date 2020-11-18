from PIL import Image
import numpy as np
import cv2 as cv

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimate_calories_from_image import compute_volume, compute_calories, compute_volume_with_grabcut, get_foreground_pixels

def test():
  # helper functions for test
  def load_image_into_numpy_array(path, w, h):
    import tensorflow as tf
    from six import BytesIO
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if(path.startswith('http')):
      response = urlopen(path)
      image_data = response.read()
      image_data = BytesIO(image_data)
      image = Image.open(image_data)
    else:
      image_data = tf.io.gfile.GFile(path, 'rb').read()
      image = Image.open(BytesIO(image_data))

    bsize = [w, h]
    image.thumbnail(bsize, Image.ANTIALIAS)
    (im_width, im_height) = image.size
    print("Resized Width {}  Height {}".format(im_width, im_height))
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)

  from matplotlib.image import imread
  
  # create numpy image
  w, h = (800, 600)
  side_image_np = load_image_into_numpy_array('scripts/testing/unittests/grabcut-result-apple.png', w, h)[0]
  top_image_np = load_image_into_numpy_array('scripts/testing/unittests/grabcut-result-apple.png', w, h)[0]

  # y_min, x_min, y_max, x_max
  food_side_bounding_box = (0.1, 0.1, 0.6, 0.6)
  food_top_bounding_box = (0.1, 0.1, 0.9, 0.9)

  coin_side_bounding_box = (0.1, 0.1, 0.2, 0.2)
  coin_top_bounding_box = (0.1, 0.1, 0.2, 0.2)

  food_label = 'apple'

  # get_foreground_pixels(side_image_np[0], (10,10,60,60))
  # compute_volume_with_grabcut(side_image_np, top_image_np, (300, 200, 200, 200), (300, 200, 200, 200), 5, 5, 'apple')
  volume = compute_volume(side_image_np, top_image_np, food_side_bounding_box, food_top_bounding_box, coin_side_bounding_box, coin_top_bounding_box, food_label)
  compute_calories(volume)

test()