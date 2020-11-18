import argparse

from detector_core import ObjectDetectorAI, load_image_into_numpy_array, get_bounding_boxes, visualize_result
from estimate_calories_from_image import compute_volume, compute_calories, FOOD_LABELS


def get_calories(ai, image_path_s, image_path_t):
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

  food_calorie_list = {}
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
    return food_calorie_list

  # Can't handle multiple of the same object in one screen
  # Load each list of bounding boxes and map to labels
  for box in bounding_boxes_s:
    label = box_to_label_s[box]
    label_to_bounding_box_s[label] = box
  
  for box in bounding_boxes_t:
    label = box_to_label_t[box]
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
      food_calorie_list[label] = compute_calories(label, volume)

  return food_calorie_list

def main():
  # Parse input
  parser = argparse.ArgumentParser(description="Run inference or compute calories.",
                                   formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
    '-s', '--sideViewImagePath',
    help='Path to image side view.',
    type=str,
    default='data/apple001S(1).jpg' 
  )
  parser.add_argument(
    '-t', '--topViewImagePath',
    help='Path to image top view.',
    type=str,
    default='data/apple001T(1).jpg' 
  )

  args = parser.parse_args()

  image_path_s = args.sideViewImagePath
  image_path_t = args.topViewImagePath
  
  # Load model
  ai = ObjectDetectorAI('ssd-640')

  # Perform inference, compute calories and visualize the result
  food_calories_list = get_calories(ai, image_path_s, image_path_t)
  print(f'Results: {food_calories_list}')

if __name__ == '__main__':
  main()