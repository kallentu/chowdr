# chowdr
Food recognition and volume estimation to produce caloric data. 

chowdr will use a trained model on the base Faster R-CNN TensorFlow model to detect different food objects. Using this data, it estimates the volume of the food
from the top and side views, then calculates the caloric amount of the food.


## Set Up

1. Clone the ECUST food dataset into the directory above chowdr.
```bash
git clone https://github.com/Liang-yc/ECUSTFD-resized-.git ../
```

## Set Up GCP

This guide assumes you have a GCP account created and is made specifically for the chowdr project.

1.  `python3 -m pip install tensorflow_cloud` and install any other relevant dependencies.

2.  Create a GCP Project called `chowder-bucket` using this 
[guide](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

3.  Set your project as the active project, using the ID found [here](https://console.cloud.google.com/).

Optionally, using:
```
export PROJECT_ID=<your-project-id>
gcloud config set project $PROJECT_ID
```

4.  Enable [AI Platform Services](https://console.cloud.google.com/apis/library/ml.googleapis.com?q=ai&id=2019ad60-5180-4001-9a93-e99871e3207b) for your project.

5.  Enable [Cloud Build API](https://console.cloud.google.com/apis/library/cloudbuild.googleapis.com?q=cloud%20build&id=9472915e-c82c-4bef-8a6a-34c81e5aebcc) for your project.

6.  Create a service account using this [guide](https://cloud.google.com/iam/docs/creating-managing-service-accounts).

7.  Generate an authentication key for the service account. Download it and create the key environment variable.

```
export GOOGLE_APPLICATION_CREDENTIALS=~/<key-name>.json
```

Now you are ready to train/test using GCP. Look at the scripts below to help you.

## Pre-processing Scripts

All scripts are run from the main directory.

### Splitting images into k-folds for cross validation.

Will split the ECUSTFD images into `train` and `test` folders in the output directory (`/workspace/`) for
cross validation training on k-folds.
Use `python3 scripts/preprocessing/kfold_partition_dataset.py -h` for parameter usage.

``` bash
python3 scripts/preprocessing/kfold_partition_dataset.py -i <input-dir> -o <output-dir> -k <k-folds> [-s <random-seed>] [-x]
```

Example:

``` bash
python3 scripts/preprocessing/kfold_partition_dataset.py -i ../ECUSTFD-resized-/JPEGImages/ -o workspace/ -k 5
```

### Create TensorFlow Records

Converts XML files to TFRecord files in preparation for training.

Use `python3 scripts/preprocessing/generate_tfrecord.py -h` for parameter usage.

Example:

```bash
python scripts/preprocessing/generate_tfrecord.py -x workspace/train_0fold -l workspace/train_0fold/annotations/label_map.pbtxt -o workspace/train_0fold/annotations/train0.record
```

## Training Scripts

All scripts are run from the main directory.

### Training a Custom Model

Initiates a new custom object detection training job for the base-model in `model_dir`.

Use `python3 scripts/training/create_run_model.py -h` for parameter usage.

Example:
```bash
python3 scripts/training/create_run_model.py --model_dir=models/faster_rcnn_0 --pipeline_config_path=models/faster_rcnn_0/pipeline.config
```

### Running test training on GCP

Follow the instructions above to set up GCP, then run this script from the main directory.

`scripts/gcp/requirements.txt` is necessary for the running of this script. It is a list of python
packages that the model depends on.

Change the `GCP_BUCKET` variable in the script to match your bucket name if it is not already `chowdr-bucket`.

```bash
python3 scripts/gcp/test_gcp_train.py
```

## Testing Scripts

All scripts are run from the main directory.

### Running an Evaluation on Trained Model
Evaluates how well the model performs in detecting objects in the test dataset that is configured in `pipeline.config`.

Use `python3 scripts/training/create_run_model.py -h` for parameter usage.

Example:
```bash
python3 scripts/training/create_run_model.py --model_dir=models/faster_rcnn_0 --pipeline_config_path=models/faster_rcnn_0/pipeline.config
```

### Export an Object Detection Model for Inference.

Prepares an object detection tensorflow graph for inference using model
configuration and a trained checkpoint. Outputs associated checkpoint files,
a SavedModel, and a copy of the model config.

Use `python3 scripts/testing/export_model.py -h` for parameter usage.

Example:
```bash
python3 scripts/testing/export_model.py --input_type image_tensor --pipeline_config_path models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config --trained_checkpoint_dir models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/ --output_directory exported-models/faster_rcnn_
```

### Running Test Images on Pre-Trained Object Detectors

Use and configure this script to run a sample, pre-trained object detection model on test images.

```bash
python3 scripts/testing/test_sample.py
```

### Running The Calorie Detector With A Test Image

Run this script to use our custom object detection model on a specified pair of test images. `-s` defaults to `data/apple001S(1).JPG` and `-t` defaults to `data/apple001T(1).JPG`.

Specify two input images:
```bash
python3 scripts/testing/calorie_detector.py -s <side view image path> -t <top view image path>
```

Use `python3 scripts/testing/calorie_detector.py -h` for parameter usage.

Example:
```bash
python3 scripts/testing/calorie_detector.py -s "data/apple001S(1).JPG" -t "data/apple001T(1).JPG"
```

**Output:**

<img width="1073" alt="command line output" src="https://user-images.githubusercontent.com/14525856/99302653-aad23e80-281d-11eb-9244-b4ad224282fc.png">

**Original Images**

<img width="300" alt="apples" src="https://user-images.githubusercontent.com/14525856/99470900-8d7f9c00-2913-11eb-8b35-5069fb6a6955.jpg">
<img width="300" alt="applet" src="https://user-images.githubusercontent.com/14525856/99470902-8f495f80-2913-11eb-9a45-8885b7cb3fc4.jpg">

**Detected Images**

<img width="300" alt="apples" src="https://user-images.githubusercontent.com/14525856/99315353-87fd5580-2830-11eb-9979-0f88fa7f3cc4.png">
<img width="300" alt="applet" src="https://user-images.githubusercontent.com/14525856/99315359-89c71900-2830-11eb-8691-3d61a30eb847.png">

**Grabcut Results:**

<img width="300" alt="grabcut-result-apple-s" src="https://user-images.githubusercontent.com/14525856/99315454-aebb8c00-2830-11eb-8dc3-689c997c3b0c.png">
<img width="300" alt="grabcut-result-apple-t" src="https://user-images.githubusercontent.com/14525856/99315459-af542280-2830-11eb-87cf-96358e902624.png">

### Running Analysis Tools

Use this script to perform analysis, estimations, error and beta computations across the entire ECUSTF dataset.

_Prerequisite:_ You must clone ECUST dataset into `./dataset/`.
```bash
git clone git@github.com:Liang-yc/ECUSTFD-resized-.git ./dataset/
```

```bash
python3 scripts/testing/compute_betas.py
```
