# CONSTANTS
PROJ_DIR = "./"
DATASET_DIR = PROJ_DIR + "first20/"
TRAIN_DIR = DATASET_DIR + "images/"
LABELS_DIR_KITTI = DATASET_DIR + "training/label_2/"
LABELS_DIR_YOLO = DATASET_DIR + "labels/"
NUM_LABELED = 7481  # Total number of labelled images
NUM_LABELS = 14     # Number of labels per image (ignoring score)
NUM_TRAIN = 5985    # Train: 80% of images SUGGESTED: 6500
NUM_VAL = 748       # Val:   10% of images SUGGESTED: 500
NUM_TEST = 748      # Test:  10% of images SUGGESTED: 481
MIN_IMG_WIDTH = 1224
MIN_IMG_HEIGHT = 370
IMG_WIDTH = MIN_IMG_WIDTH
IMG_HEIGHT = MIN_IMG_HEIGHT
IMG_DEPTH = 3