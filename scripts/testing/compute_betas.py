import csv
import json

from detector_core import ObjectDetectorAI, load_image_into_numpy_array, get_bounding_boxes, visualize_result
from estimate_calories_from_image import compute_volume, compute_calories, FOOD_LABELS


FOOD_LABEL_NAMES = [
  'apple',
  'banana',
  'bread',
  'bun',
  'doughnut',
  'egg',
  'fired dough twist',
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
]

ROOT_DIR = 'scripts/testing/data/'

def get_volume_measurements(ai, image_path_s, image_path_t):
  # Load numpy arrays
  image_np_s = load_image_into_numpy_array(image_path_s)
  image_np_t = load_image_into_numpy_array(image_path_t)

  # Run inference
  inference_results_s = ai.run_inference(image_np_s)
  inference_results_t = ai.run_inference(image_np_t)

  bounding_boxes_s, box_to_label_s = get_bounding_boxes(inference_results_s)
  bounding_boxes_t, box_to_label_t = get_bounding_boxes(inference_results_t)

  visualize_result(image_np_s, inference_results_s, image_path_s + '-detected.png')
  visualize_result(image_np_t, inference_results_t, image_path_t + '-detected.png')

  food_volume_list = {}
  coin_bounding_box_s = None
  coin_bounding_box_t = None
  label_to_bounding_box_s = {}
  label_to_bounding_box_t = {}

  # Get the bounding box for coins
  for box in bounding_boxes_s:
    if box_to_label_s[box] == 'coin':
      coin_bounding_box_s = box
      box_to_label_s.pop(box)
      bounding_boxes_s.remove(box)
      break

  for box in bounding_boxes_t:
    if box_to_label_t[box] == 'coin':
      coin_bounding_box_t = box
      box_to_label_t.pop(box)
      bounding_boxes_t.remove(box)
      break
    
  if not coin_bounding_box_s or not coin_bounding_box_t:
    print('Could not detect the coin')
    return food_volume_list

  # Can't handle multiple of the same object in one screen
  # Load each list of bounding boxes and map to labels
  for box in bounding_boxes_s:
    label = box_to_label_s[box].replace('_',' ')
    label_to_bounding_box_s[label] = box
  
  for box in bounding_boxes_t:
    label = box_to_label_t[box].replace('_',' ')
    label_to_bounding_box_t[label] = box

  # If food is contained in both image views, compute the calories
  for label in label_to_bounding_box_s:
    if label in label_to_bounding_box_t:
      food_bounding_box_s = label_to_bounding_box_s[label]
      food_bounding_box_t = label_to_bounding_box_t[label]
      volume = compute_volume(image_np_s[0], 
                              image_np_t[0], 
                              food_bounding_box_s, 
                              food_bounding_box_t, 
                              coin_bounding_box_s, 
                              coin_bounding_box_t, 
                              label)
      food_volume_list[label] = volume
  return food_volume_list

def get_actual_volumes():
  label_to_filenames = dict()
  label_to_filename_volume = dict()
  label_to_average = dict()

  for label in FOOD_LABEL_NAMES:
    label_file_path = label.replace(' ', '_')
    with open(f'data/beta/volumes/{label_file_path}-Table 1.csv') as csv_file:
      items = list(csv.DictReader(csv_file))

      total_volume = 0
      label_to_filenames[label] = []
      label_to_filename_volume[label] = {}
      for row in items:
        label_to_filenames[label].append(row['id'])
        volume = int(row['volume(mm^3)'])
        label_to_filename_volume[label][row['id']] = volume
        total_volume += volume
      label_to_average[label] = total_volume / len(items)

  return label_to_filenames, label_to_filename_volume, label_to_average

# actual_volume: get average volumes for each food from the table
# detected_volume: haha thanget average volume from all our detection work
# get beta value from these two averages
# beta = actual_volume / detected_voume 
# dict: label : beta 

def estimate_volumes(ai, label_to_filenames):
  root_dir = 'dataset/ECUSTFD-resized-/JPEGImages/'

  # {label : {filename : volume}}
  label_to_filename_volume = dict()
  # {label : average volume}
  label_to_average = dict()
  
  for label in label_to_filenames:
    total_volume = 0
    label_to_filename_volume[label] = {}
    detected_images = 0

    for filename in label_to_filenames[label]:
      filename_file_path = filename.replace(' ', '_')
      image_path_s = f'{root_dir}{filename_file_path}S(1).JPG'
      image_path_t = f'{root_dir}{filename_file_path}T(1).JPG'
      print(f'Starting volume estimation: {label}, {image_path_s}, {image_path_t}')

      label_to_volume = get_volume_measurements(ai, image_path_s, image_path_t)

      print(f'Volume estimation         : {label}, {label_to_volume}')

      if label in label_to_volume:
        volume = label_to_volume[label]
        label_to_filename_volume[label][filename] = volume
        total_volume += volume
        detected_images += 1
    
    label_to_average[label] = total_volume / detected_images
  
  print(label_to_filename_volume)
  print(label_to_average)
  return label_to_filename_volume, label_to_average

def output_measurements():
  ai = ObjectDetectorAI('ssd-640')

  label_to_filenames, real_label_to_filename_volume, real_label_to_average = get_actual_volumes()
  with open(f'{ROOT_DIR}real_label_to_filename_volume.txt', 'w') as outfile:
    json.dump(real_label_to_filename_volume, outfile)
  with open(f'{ROOT_DIR}real_label_to_average.txt', 'w') as outfile:
    json.dump(real_label_to_average, outfile)

  measured_label_to_filename_volume, measured_label_to_average = estimate_volumes(ai, label_to_filenames)
  with open(f'{ROOT_DIR}measured_label_to_filename_volume_beta.txt', 'w') as outfile:
    json.dump(measured_label_to_filename_volume, outfile)
  with open(f'{ROOT_DIR}measured_label_to_average_beta.txt', 'w') as outfile:
    json.dump(measured_label_to_average, outfile)

def print_volumes():
  real_averages = None
  measured_averages = None
  label_to_beta = {}

  with open(f'{ROOT_DIR}real_label_to_average.txt') as infile:
    real_averages = json.load(infile)
  with open(f'{ROOT_DIR}measured_label_to_average.txt') as infile:
    measured_averages = json.load(infile)

  for food_label in real_averages:
    label_to_beta[food_label] = round(real_averages[food_label]/measured_averages[food_label] * 10000) / 10000.0
  
  print('labels:')
  for f in measured_averages:
    print(f)
  print('real_averages:')
  for f in real_averages:
    print(measured_averages[f])
  print(f'measured_averages:')
  for f in measured_averages:
    print(measured_averages[f])

def test_with_beta():
  ai = ObjectDetectorAI('ssd-640')

  label_to_filenames, real_label_to_filename_volume, _ = get_actual_volumes()

  # for each food label, pick the first file item and run it through detection
  # return the volume (using the new beta values)

  root_dir = 'dataset/ECUSTFD-resized-/JPEGImages/'

  # {label : {filename : volume}}
  label_to_filename_error = dict()
  # {label : average volume}
  label_to_average_error = dict()
  
  for label in label_to_filenames:
    total_error = 0
    label_to_filename_error[label] = {}
    detected_images = 0

    for filename in label_to_filenames[label]:
      real_volume = real_label_to_filename_volume[label][filename]
      filename_file_path = filename.replace(' ', '_')
      image_path_s = f'{root_dir}{filename_file_path}S(1).JPG'
      image_path_t = f'{root_dir}{filename_file_path}T(1).JPG'
      print(f'Starting volume estimation: {label}, {image_path_s}, {image_path_t}')

      label_to_volume = get_volume_measurements(ai, image_path_s, image_path_t)

      print(f'Volume estimation         : {label}, {label_to_volume}')

      if label in label_to_volume:
        volume = label_to_volume[label]
        error = (real_volume - volume) / real_volume
        label_to_filename_error[label][filename] = error
        total_error += abs(error)
        detected_images += 1
      
      label_to_average_error[label] = total_error / detected_images
  
  print(label_to_filename_error)
  print(label_to_average_error)
  for label in label_to_average_error:
    print(label)
  for label in label_to_average_error:
    print(label_to_average_error[label])


  with open(f'{ROOT_DIR}label_to_filename_error.txt', 'w') as outfile:
    json.dump(label_to_filename_error, outfile)
  with open(f'{ROOT_DIR}label_to_average_error.txt', 'w') as outfile:
    json.dump(label_to_average_error, outfile)

  return label_to_filename_error, label_to_average_error

def compute_average_errors():
  with open(f'{ROOT_DIR}label_to_filename_error.txt') as infile:
    measured_errors = json.load(infile)

    avg_errors = {}
    for label in measured_errors:
      total_error = 0
      for name in measured_errors[label]:
        total_error += abs(measured_errors[label][name])
      avg_errors[label] = total_error / len(measured_errors[label])
    
    print('labels:')
    for f in avg_errors:
      print(f)
    print('averages:')
    for f in avg_errors:
      print(avg_errors[f])

def compute_errors():
  with open(f'{ROOT_DIR}real_label_to_filename_volume.txt') as infile:
    real_volumes = json.load(infile)

    with open(f'{ROOT_DIR}measured_label_to_filename_volume_beta.txt') as infile:
      measured_volumes = json.load(infile)

      avg_errors = {}
      for label in real_volumes:
        items = 0
        total_error = 0
        for label_file_name in real_volumes[label]:
          if label_file_name in measured_volumes[label]:
            items += 1
            real_v = real_volumes[label][label_file_name]
            measured_v = measured_volumes[label][label_file_name] 
            total_error += abs((real_v - measured_v) / real_v)
        
        avg_errors[label] = total_error / items
      
      print('labels:')
      for f in avg_errors:
        print(f)
      print('averages:')
      for f in avg_errors:
        print(avg_errors[f])

def main():
  # output_measurements()
  compute_errors()
  # print_volumes()
  # test_with_beta()
  # compute_average_errors()

if __name__ == '__main__':
  main()
