""" usage: kfold_partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-k KFOLDS] [-x] [-s SEED]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -k KFOLDS, --kfolds KFOLDS
                        The number of folds over the total number of images, used for k-fold cross validation. The default is 10.
  -x, --xml             Set this flag if you want the xml annotation files to be processed and copied over.
  -s, --seed SEED       Set the seed for shuffling randomization.
"""
import os
import re
from shutil import copyfile
import argparse
import math
import random

# Imports for k-fold cross-validation
from sklearn.model_selection import KFold
import numpy as np

def iterate_dir(source, dest, k, copy_xml):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')


    # Shuffle images first so we get a good test.
    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.JPG|.jpg|.jpeg|.png)$', f)]
    random.shuffle(images)

    if images == None or len(images) == 0:
      print("Error: No images found.")
      exit(1)

    # Fold data set into [k] folds.
    # For each fold, use 1/[k] as validation data and [k]-1/[k] as test data.
    np_images = np.array(images)
    kf = KFold(n_splits=k, shuffle=False)
    print(kf)

    # Count keeps track of which fold we are in.
    kfold_count = 0
    for train_index, test_index in kf.split(np_images):
      print(f"Current fold: {kfold_count}")
      print(f"Size of training data: {train_index.size} Size of testing data: {test_index.size}")

      # Make directories.
      train_dir = os.path.join(dest, f'train_{kfold_count}fold')
      test_dir = os.path.join(dest, f'test_{kfold_count}fold')

      if not os.path.exists(train_dir):
        os.makedirs(train_dir)
      if not os.path.exists(test_dir):
        os.makedirs(test_dir)

      # Copy training images.
      train_images = np.take(np_images, train_index)
      for filename in train_images:
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(source, "../Annotations", xml_filename),
                     os.path.join(train_dir, xml_filename))

      # Copy testing images.
      test_images = np.take(np_images, test_index)
      for filename in test_images:
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(source, "../Annotations", xml_filename),
                     os.path.join(test_dir,xml_filename))

      # Update count.
      kfold_count += 1

# Example command: python3 scripts/preprocessing/kfold_partition_dataset.py -i ../ECUSTFD-resized-/JPEGImages/ -o workspace/ -k 5
def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets based on k-folds.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-k', '--kfolds',
        help='The number of folds over the total number of images used for cross-validation. The default is 10.',
        default=10,
        type=int)
    parser.add_argument(
        '-x', '--xml',
        help='Set this flag if you want the xml annotation files to be processed and copied over.',
        action='store_true'
    )
    parser.add_argument(
        '-s', '--seed',
        help='Set the seed for shuffling randomization.',
        default=10,
        type=int)
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    if args.imageDir is None:
      print("Error: Need image directory.")
      sys.exit(1)

    # Seed random
    random.seed(args.seed)
    print(f"Random seed is {args.seed}.")

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.kfolds, args.xml)


if __name__ == '__main__':
    main()
