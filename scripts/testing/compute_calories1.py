
from PIL import Image
import numpy as np
import cv2 as cv
import math


COIN_WIDTH = 2.5

# value can be 'ellipsoid', 'column', 'irregular'
FOOD_SHAPE = {
  'apple' : 'ellipsoid',
  'banana' : 'irregular',
  'bread' : 'column',
  'bun' : 'irregular',
  'doughnut' : 'irregular',
  'egg' : 'ellipsoid',
  'fried dough twist' : 'irregular',
  'grape' : 'column',
  'lemon' : 'ellipsoid',
  'litchi' : 'irregular',
  'mango' : 'irregular',
  'mooncake' : 'column',
  'orange' : 'ellipsoid',
  'peach' : 'ellipsoid',
  'pear' : 'irregular',
  'plum' : 'ellipsoid',
  'qiwi' : 'ellipsoid',
  'sachima' : 'column',
  'tomato' : 'ellipsoid'
}

# TODO: Populate these values with empirical data that we come up with
FOOD_BETA = {
  'apple' : 1.0,
  'banana' : 1.0,
  'bread' : 1.0,
  'bun' : 1.0,
  'doughnut' : 1.0,
  'egg' : 1.0,
  'fried dough twist' : 1.0,
  'grape' : 1.0,
  'lemon' : 1.0,
  'litchi' : 1.0,
  'mango' : 1.0,
  'mooncake' : 1.0,
  'orange' : 1.0,
  'peach' : 1.0,
  'pear' : 1.0,
  'plum' : 1.0,
  'qiwi' : 1.0,
  'sachima' : 1.0,
  'tomato' : 1.0
}

FOOD_CALORIE = { 
  'apple' : 0.4056,
  'banana' : 0.8099,
  'bread' : 0.567,
  'bun' : 0.7582,
  'doughnut' : 1.3454,
  'egg' : 1.4729,
  'fired_dough_twist' : 14.0128,
  'grape' : 0.6693,
  'lemon' : 0.2784,
  'litchi' : 0.66,
  'mango' : 0.642,
  'mooncake' : 18.0768,
  'orange' : 0.567,
  'peach' : 0.5472,
  'pear' : 0.3978,
  'plum' : 0.4646,
  'qiwi' : 0.5917,
  'sachima' : 4.719,
  'tomato' : 0.2646 
}

# Returns object width and height in pixels
def get_dimensions(bounding_box, width, height):
  ymin, xmin, ymax, xmax = bounding_box
  object_width_px = (xmax - xmin) * width
  object_height_px = (ymax - ymin) * height
  return object_width_px, object_height_px

# Take in 2 images: Side view, Top view
# Variables Required
# Side view: width, height of coin
# Top view: width, height of coin


# Grabcut pixel counts
# - Sideview from the bottom layer to the top layer
# - Topview from the bottom layer to the top layer s_T
# - Sideview max length of sideview pixels + the sideview from bottom layer to the top layer

def np_to_cv(image_np):
  img = Image.fromarray(image_np, 'RGB')
  # img.save('np_img.png')

  # convert Image to cv
  imgcv = np.asarray(img)[:,:,::-1].copy()
  imgcv = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

  # output cv
  # cv.imwrite('cv_img.png', imgcv)
  # cv.imshow("image", imgcv)
  return imgcv

# Uses grabcut to get an array of foreground pixels for the given numpy image array
def get_foreground_pixels(image_np, bounding_box):

  # convert np array to cv version
  img = np_to_cv(image_np)

  mask = np.zeros(img.shape[:2],np.uint8)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)

  cv.grabCut(img,mask,bounding_box,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  img = img*mask2[:,:,np.newaxis]
  pil_img = Image.fromarray((img * 255).astype(np.uint8))

  # pil_img.save('foreground.png')

  # convert Image to black & white cv (1 colour channel)
  pixel_image = np.asarray(pil_img)[:,:,::-1].copy()
  pixel_image = cv.cvtColor(np.asarray(pil_img), cv.COLOR_BGR2GRAY)

  coloured_pixels = cv.countNonZero(pixel_image)
  print(f'coloured pixels: {coloured_pixels}')

  return pixel_image

def compute_volume_with_grabcut(side_image_np, 
                                top_image_np, 
                                food_side_bounds,
                                food_top_bounds,
                                side_scaling_factor, 
                                top_scaling_factor, 
                                food_label):
  side_image_foreground_pixels = get_foreground_pixels(side_image_np.copy(), food_side_bounds)
  top_image_foreground_pixels = get_foreground_pixels(top_image_np.copy(), food_top_bounds)

  food_beta = FOOD_BETA[food_label]

  def calc_L_s_k_squared():
    squared_pixel_rows_sum = 0
    for row in side_image_foreground_pixels:
        row_sum = 0
        for pixel_val in row:
            if pixel_val != 0:
                row_sum = row_sum + 1
        squared_pixel_rows_sum = squared_pixel_rows_sum + row_sum ** 2
    return squared_pixel_rows_sum

  def calc_s_T():
    s_T = 0
    H_s = top_image_foreground_pixels.shape[0]
    for row in top_image_foreground_pixels:
        row_sum = 0
        for pixel_val in row:
            if pixel_val != 0:
                row_sum = row_sum + 1
        s_T = s_T + row_sum
    return s_T
  
  def calc_L_s_MAX_squared():
    max_row = 0
    for row in side_image_foreground_pixels:
        row_total = 0
        for pixel_val in row:
          if pixel_val != 0:
            row_total = row_total + 1
        if row_total > max_row:
          max_row = row_total

    ret = 0
    for row in side_image_foreground_pixels:
        row_sum = 0
        for pixel_val in row:
            if pixel_val != 0:
                row_sum = row_sum + 1
        ret = ret + ((row_sum/max_row) ** 2)
    return ret

  if FOOD_SHAPE[food_label] == 'ellipsoid':
    # Compute squared pixel rows sum term (sum(Lks^2))
    return food_beta * math.pi / 4 * calc_L_s_k_squared() * (side_scaling_factor ** 3)
  elif FOOD_SHAPE[food_label] == 'column':
    # Compute pixel rows sum term (s_T = sum(L^k_T))
    H_s = top_image_foreground_pixels.shape[0]
    return food_beta * (calc_s_T() * top_scaling_factor ** 2) * (H_s * side_scaling_factor)
  elif FOOD_SHAPE[food_label] == 'irregular':
    # TODO: Determine volume of the irregular shape
    return food_beta * (calc_s_T() * top_scaling_factor ** 2) * calc_L_s_MAX_squared() * side_scaling_factor
  else:
    return 0.0

def get_measurements(image_np, coin_bounding_box, food_bounding_box):
  image_height_px, image_width_px, _ = image_np.shape
  coin_w_px, coin_h_px = get_dimensions(coin_bounding_box, image_width_px, image_height_px)
  scaling_factor = COIN_WIDTH / ((coin_w_px + coin_h_px) / 2)
  food_w_px, food_h_px = get_dimensions(food_bounding_box, image_width_px, image_height_px)
  return image_width_px, image_height_px, scaling_factor, food_w_px, food_h_px

def compute_calories(side_image_np, 
                     top_image_np, 
                     food_side_bounding_box, 
                     food_top_bounding_box, 
                     coin_side_bounding_box, 
                     coin_top_bounding_box,
                     food_label):

  # compute scaling factor for coin side and top
  side_image_width_px, side_image_height_px, side_scaling_factor, food_side_width_px, food_side_height_px = \
    get_measurements(side_image_np, coin_side_bounding_box, food_side_bounding_box)
  top_image_width_px, top_image_height_px, top_scaling_factor, food_top_width_px, food_top_height_px = \
    get_measurements(top_image_np, coin_top_bounding_box, food_top_bounding_box)                                                                             
  
  # compute bounds for foods
  food_side_bounds = (int(food_side_bounding_box[0] * side_image_width_px), \
                      int(food_side_bounding_box[1] * side_image_height_px), \
                      int(food_side_width_px), \
                      int(food_side_height_px))

  food_top_bounds = (int(food_top_bounding_box[0] * top_image_width_px), \
                     int(food_top_bounding_box[1] * top_image_height_px), \
                     int(food_top_width_px), \
                     int(food_top_height_px))

  # compute volume
  volume = compute_volume_with_grabcut(side_image_np, 
                                       top_image_np, 
                                       food_side_bounds, 
                                       food_top_bounds, 
                                       side_scaling_factor, 
                                       top_scaling_factor,
                                       food_label)

  # compute calories
  return volume * FOOD_CALORIE[food_label]



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
  w, h = 800, 600
  # side_image_np = np.zeros((h, w, 3), dtype=np.uint8)
  # side_image_np[0:50, 0:50] = [255, 0, 0] # red patch in upper left
  
  # top_image_np = np.zeros((h, w, 3), dtype=np.uint8)
  # top_image_np[50:99, 50:99] = [255, 0, 0] # red patch in bottom right

  side_image_np = load_image_into_numpy_array('scripts/testing/testdata/grabcut-result-apple.png', w, h)[0]
  top_image_np = load_image_into_numpy_array('scripts/testing/testdata/grabcut-result-apple.png', w, h)[0]



  food_side_bounding_box = (0.1, 0.1, 0.6, 0.6)
  food_top_bounding_box = (0.1, 0.1, 0.9, 0.9)

  coin_side_bounding_box = (0.1, 0.1, 0.2, 0.2)
  coin_top_bounding_box = (0.1, 0.1, 0.2, 0.2)

  food_label = 'apple'

  # get_foreground_pixels(side_image_np[0], (10,10,60,60))
  # compute_volume_with_grabcut(side_image_np, top_image_np, (300, 200, 200, 200), (300, 200, 200, 200), 5, 5, 'apple')
  compute_calories(side_image_np, top_image_np, food_side_bounding_box, food_top_bounding_box, coin_side_bounding_box, coin_top_bounding_box, food_label)

test()