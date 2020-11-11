# chowdr
Food recognition and volume estimation to produce caloric data.

## Set Up

1. Clone the ECUST food dataset into the directory above chowdr.
```bash
git clone https://github.com/Liang-yc/ECUSTFD-resized-.git ../
```

## Scripts

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
