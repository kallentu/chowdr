# chowdr
Food recognition and volume estimation to produce caloric data.

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

## Scripts

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

### Running test training on GCP

Follow the instructions above to set up GCP, then run this script from the main directory.

`scripts/gcp/requirements.txt` is necessary for the running of this script. It is a list of python
packages that the model depends on.

Change the `GCP_BUCKET` variable in the script to match your bucket name if it is not already `chowdr-bucket`.

```bash
python3 scripts/gcp/test_gcp_train.py
```

### Running Test Images on Pre-Trained Object Detectors

Use and configure this script to run a sample, pre-trained object detection model on test images.

```bash
python3 scripts/test_sample.py
```

### Running Apple Test Image on Custom Object Detector

Use and configure this script to run a custom object detection model on the test image `apple001S(1).JPG` in the ECUSTFD dataset.

```bash
python3 scripts/test_apple.py
```
