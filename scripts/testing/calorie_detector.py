import os
import pathlib
import json
import argparse

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
import six
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

from estimate_calories_from_image import compute_calories, FOOD_LABELS

# Set up tensorflow
tf.get_logger().setLevel('ERROR')

# Global declarations
TRAINED_MODELS = {
  'ssd-640' : "workspace/training_demo/exported-models/faster_fold1_boi/saved_model/"
}

# Labels for image objects
PATH_TO_LABELS = 'workspace/training_demo/annotations/label_map.pbtxt'

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

## AI Class
# Allows up to load up the model model once be able to call inference multiple times
class ObjectDetectorAI:
  def __init__(self, selected_model): # @param ['ssd-640']
    global TRAINED_MODELS

    model_path = TRAINED_MODELS[selected_model]
    print('Loading model...')
    self.hub_model = tf.saved_model.load(model_path)
    print('Model loaded!')

  def run_inference(self, image_np):
    return self.hub_model(image_np)

def load_image_into_numpy_array(path):
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

  size = [256, 256]
  bsize = [640, 640]
  image.thumbnail(bsize, Image.ANTIALIAS)
  (im_width, im_height) = image.size
  print("Resized Width {}  Height {}".format(im_width, im_height))
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)

def visualize_result(image_np, results, output_file):
  image_np_with_detections = image_np.copy()

  # different object detection models have additional results
  # all of them are explained in the documentation
  result = {key:value.numpy() for key,value in results.items()}

  label_id_offset = 0

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  # Use keypoints if available in detections
  keypoints, keypoint_scores = None, None
  if 'detection_keypoints' in result:
    keypoints = result['detection_keypoints'][0]
    keypoint_scores = result['detection_keypoint_scores'][0]
  
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

  plt.figure(figsize=(24,32))
  plt.imshow(image_np_with_detections[0])
  plt.savefig(output_file)

# Returning all bounding boxes where the confidence is > confidence_threshold 
def get_bounding_boxes(results, confidence_threshold = 0.5):
  # different object detection models have additional results
  # all of them are explained in the documentation
  result = {key:value.numpy() for key,value in results.items()}

  # Get information for the objects
  result['detection_boxes'][0].shape
  boxes = result['detection_boxes'][0]
  scores = result['detection_scores'][0]
  classes = result['detection_classes'][0]
  
  # Get the boxes and box labels
  box_list = []
  box_to_label = {}

  for i in range(boxes.shape[0]):
    if scores is None or scores[i] > confidence_threshold:
      box = tuple(boxes[i].tolist())  
      box_list.append(box)

      display_str = ''
      if classes[i] in six.viewkeys(category_index):
        class_name = category_index[classes[i]]['name']
      else:
        class_name = 'N/A'
      display_str = str(class_name)
      box_to_label[box] = display_str
  
  return box_list, box_to_label

def get_calories(ai, image_path_s, image_path_t):
  # Load numpy arrays
  image_np_s = load_image_into_numpy_array(image_path_s)
  image_np_t = load_image_into_numpy_array(image_path_t)

  # Run inference
  inference_results_s = ai.run_inference(image_np_s)
  inference_results_t = ai.run_inference(image_np_t)

  bounding_boxes_s, box_to_label_s = get_bounding_boxes(inference_results_s)
  bounding_boxes_t, box_to_label_t = get_bounding_boxes(inference_results_t)
  print(box_to_label_s)
  print(box_to_label_t)

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
      food_calorie_list[label] = compute_calories(image_np_s[0], 
                                                  image_np_t[0], 
                                                  food_bounding_box_s, 
                                                  food_bounding_box_t, 
                                                  coin_bounding_box_s, 
                                                  coin_bounding_box_t, 
                                                  label)
  
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