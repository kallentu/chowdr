import math
from usda import UsdaClient

client = UsdaClient('NIgQPVwVjH8xEkL3hzcBYZqJh4aW1lT89owwvmwT')

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

# TODO: Add real calorie/cm^3 for each item or use USDA
FOOD_CALORIE = {
    'apple' : 0.21,
    'pear' : 3,
    'banana' : 3,
    'bread' : 3,
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
# TODO: Use pixel counts for possibly better accuracy
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

# TODO: Use grabcut to do a pixel count to determine food volume
#def compute_volume_with_grab_cut():

# TODO: Resolve issue where we must access the USDA API through https
# def get_unit_calorie_using_usda(food_label):
#   food_id = next(client.search_foods(food_label, 1)).id
#   food_info = client.get_food_report(food_id)
#   print(food_info.nutrients)
#   print(f"USDA Food: {food_info.food.name}, Calories: {food_info.nutrient[calores].value}")
#   calories = food_info['calories']
#   return 1
  # get the calorie per cm^3 from this 

def get_calories(food_label, volume):
  return FOOD_CALORIE[food_label] * volume
  # TODO: use USDA's guide to calculate food calorie
  #  problem: calories & other nutritional facts are based on mass, not volume.
  #  and for apple, I had to manually calculate the calorie count based on a reference
  # return get_unit_calorie_using_usda(food_label) * volume

# take in a bounding box for all detected objects
# return a float value
def compute_calories(bounding_boxes, box_to_label, image_width_px, image_height_px):
  # pick the coin highest confidence %
  coin_box = None
  food_box = None
  food_label = None

  for box in bounding_boxes:
    if box_to_label[box] == 'coin':
      coin_box = box
      break
  
  if not coin_box:
    return -1

  # pick the food item with the highest confidence %
  # bounding_boxes are sorted in order of highest confidence first  
  for box in bounding_boxes:
    label = box_to_label[box]
    if label in FOOD_LABELS:
      food_box = box
      food_label = label
      break
  
  if not food_box or not food_label:
    return -1
  
  # compute the scaling factor with the dimensions of the coin and the image
  coin_width_px, coin_height_px = get_dimensions(coin_box, image_width_px, image_height_px)

  scaling_factor = COIN_WIDTH / ((coin_width_px + coin_height_px) / 2)

  # use scaling factor to determine the dimensions of the food
  food_width_px, food_height_px = get_dimensions(food_box, image_width_px, image_height_px)

  food_width = food_width_px * scaling_factor
  food_height = food_height_px * scaling_factor

  # compute volume with the dimensions of the food and type of food
  volume = compute_volume_geometrically(food_label, food_width, food_height)

  # compute calories given volume and type of food
  calories = get_calories(food_label, volume)
  return food_label, calories


def test():
  boxes_list_test = [
    (0.6185794472694397, 0.23970480263233185, 0.7193066477775574, 0.3143535256385803), 
    (0.4142831861972809, 0.3997975289821625, 0.729759693145752, 0.6652837991714478)
  ]

  box_to_label_test = {
    (0.6185794472694397, 0.23970480263233185, 0.7193066477775574, 0.3143535256385803): 'coin', 
    (0.4142831861972809, 0.3997975289821625, 0.729759693145752, 0.6652837991714478): 'apple'
  }

  cal = compute_calories(boxes_list_test, box_to_label_test, 640, 432)
  print("Calories", cal)

  if cal[0] != 'apple':
    print(f'Test Failed. Expected \'apple\' but got {cal[0]}')
  if cal[1] != 71.08249676475752:
    print(f'Test Failed. Expected 71.08249676475752 but got {cal[0]}')