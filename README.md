# chowdr
Food recognition and volume estimation to produce caloric data.

## Set Up

1. Clone the ECUST food dataset into the directory above chowdr.
```bash
git clone https://github.com/Liang-yc/ECUSTFD-resized-.git ../
```

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
