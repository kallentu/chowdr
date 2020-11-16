import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image

FOOD_LABELS = {
    'apple',
    'banana',
    'bread',
    'bun',
    'doughnut'
    'egg'
    'fried dough twist',
    'grape',
    'lemon',
    'litchi',
    'mango',
    'mooncake',
    'orange',
    'peach',
    'pear',
    'plum',
    'qiwi',
    'sachima',
    'tomato',
}

# value can be 'ellipsoid', 'column', 'irregular'
FOOD_SHAPE = {
  'apple' : 'ellipsoid',
  'pear' : 'ellipsoid',
  'banana' : 'column',
  'bread' : 'irregular',
}

# TODO: Populate these values with empirical data that we come up with
FOOD_BETA = {
  'apple' : 1.0,
  'pear' : 1.0,
  'banana' : 1.0,
  'bread' : 1.0,
}

FOOD_CALORIE = { 
  'apple':0.4056,
  'banana':0.8099,
  'bread':0.567,
  'bun':0.7582,
  'doughnut':1.3454,
  'egg':1.4729,
  'fired_dough_twist':14.0128,
  'grape':0.6693,
  'lemon':0.2784,
  'litchi':0.66,
  'mango':0.642,
  'mooncake':18.0768,
  'orange':0.567,
  'peach':0.5472,
  'pear':0.3978,
  'plum':0.4646,
  'qiwi':0.5917,
  'sachima':4.719,
  'tomato':0.2646 
}

# Real width of the coin in cm
COIN_WIDTH = 2.5

# Returns object width and height in pixels
def get_dimensions(bounding_box, width, height):
  ymin, xmin, ymax, xmax = bounding_box
  object_width_px = (xmax - xmin) * width
  object_height_px = (ymax - ymin) * height
  return object_width_px, object_height_px

# given type of object and dimensions, compute the volume (ellipsoid, cylinder, irregular) 
# return a float value
def compute_volume_geometrically(food_label, width, height):
  if FOOD_SHAPE[food_label] == 'ellipsoid':
    return FOOD_BETA[food_label] * 4/3 * math.pi * width/2 * width/2 * height/2
  elif FOOD_SHAPE[food_label] == 'column':
    length = max(width, height)
    radius = min(width, height) / 2
    return FOOD_BETA[food_label] * math.pi * radius ** 2 * length
  elif FOOD_SHAPE[food_label] == 'irregular':
    # TODO: Determine volume of the irregular shape
    return 0.0
  else:
    return 0.0


def compute_volume_with_grabcut(img, rect, food_label, alpha):
  clone_img = img.copy()

  mask = np.zeros(clone_img.shape[:2],np.uint8)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)

  cv.grabCut(clone_img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  clone_img = clone_img*mask2[:,:,np.newaxis]
  im = Image.fromarray((clone_img * 255).astype(np.uint8))
  im.save('temp.png')

  pixel_image = cv.imread('temp.png',0)
  coloured_pixel = cv.countNonZero(pixel_image)
  print(f'coloured pixels:{coloured_pixel}')


  if FOOD_SHAPE[food_label] == 'ellipsoid':
    # Compute squared pixel rows sum term (sum(Lks^2))
    squared_pixel_rows_sum = 0
    for row in pixel_image:
        row_sum = 0
        for pixel_val in row:
            if pixel_val != 0:
                row_sum = row_sum + 1
        squared_pixel_rows_sum = squared_pixel_rows_sum + ((row_sum/2) ** 2) * math.pi * (alpha ** 3)
    return FOOD_BETA[food_label] * squared_pixel_rows_sum 
  elif FOOD_SHAPE[food_label] == 'column':
    # TODO: Determine volume of column
    return 0.0
  elif FOOD_SHAPE[food_label] == 'irregular':
    # TODO: Determine volume of the irregular shape
    return 0.0
  else:
    return 0.0


def compute_calories(image_path, coin_box, food_box, food_label):
  img = cv.imread(image_path)
  image_height_px, image_width_px, _ = img.shape
  
  # compute the scaling factor with the dimensions of the coin and the image
  coin_width_px, coin_height_px = get_dimensions(coin_box, image_width_px, image_height_px)

  scaling_factor = COIN_WIDTH / ((coin_width_px + coin_height_px) / 2)

  # use scaling factor to determine the dimensions of the food
  food_width_px, food_height_px = get_dimensions(food_box, image_width_px, image_height_px)

  # compute volume with the dimensions of the food and type of food
  # food_width = food_width_px * scaling_factor
  # food_height = food_height_px * scaling_factor
  # volume = compute_volume_geometrically(food_label, food_width, food_height)

  # compute volume with grabcut & pixel counts
  bounds = (int(food_box[0] * image_width_px), int(food_box[1] * image_height_px), int(food_width_px), int(food_height_px))
  volume = compute_volume_with_grabcut(img, bounds, food_label, scaling_factor) 
  print(f'volume: {volume}')

  # compute calories given volume and type of food
  calories = FOOD_CALORIE[food_label] * volume
  return food_label, calories
