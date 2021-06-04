import logging
import logging.config
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import random
from tqdm import tqdm

from analysis.BoxplotHistogramCollector import BoxplotHistogramCollector
from analysis.SeverityVsGoodness import SeverityVsGoodness

IMG_WIDTH = 1224
IMG_HEIGHT = 370
PLT_TEXT_OFFSET = -10

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

##########################################################################################
# Helpers
##########################################################################################
def rel_to_absolute_label(annotation: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Converts YOLO annotation to (left, top, width, height)

    :param annotation: Single entry/annotation/object in a YOLO label
    :return (left, top, width, height): 
    """
    x, y, width, height = annotation[1:5]
    new_width = math.floor(width * IMG_WIDTH)
    new_height = math.floor(height * IMG_HEIGHT)
    left = math.floor(x * IMG_WIDTH) - new_width//2
    top = math.floor(y * IMG_HEIGHT) - new_height//2
    return (left, top, new_width, new_height)

def rel_to_box_coords(annotation: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Converts YOLO annotation to (left, top, right, bottom)

    :param annotation: Single entry/annotation/object in a YOLO label
    :return (left, top, right, bottom):
    """
    (left, top, new_width, new_height) = rel_to_absolute_label(annotation)
    right = left + new_width
    bottom = top + new_height
    return (left, top, right, bottom)

def clean_yolo(yolo_label: np.ndarray) -> np.ndarray:
    """
    Don't need to call this (usually), just call read_background_annotation_yolo().
    This helps in the case that there are no annotations or one annotation in a label

    :param yolo_label: YOLO label for an image
    :return yolo_label:
    """
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def read_background_annotation_yolo(yolo_label_path: str) -> np.ndarray:
    """
    Reads and cleans a YOLO label given a path.

    :param yolo_label_path: YOLO label filepath
    :return yolo_label:
    """
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 

def get_split_type(split: str) -> str:
    if split.startswith("train"): return "train"
    if split.startswith("val"): return "val"
    if split.startswith("test"): return "test"
    assert False, "Split type not recognised."
##########################################################################################
# End Helpers
##########################################################################################


##########################################################################################
# Plotting
##########################################################################################
def plot_bbox_yolo_modal_amodal_predictions_arrays(amodal_label: np.ndarray, modal_label: np.ndarray, prediction_label: np.ndarray, baseline_prediction_label: np.ndarray, image: np.ndarray, image_id: str, show_plot: bool, save_plot_dir: str):
    """
    Plots the bounding boxes for amodal original labels, modal original labels, the model of interest's predicted labels and, optionally, the baseline model's predicted labels.

    :param amodal_label: Numpy array of the amodal label
    :param modal_label: Numpy array of the modal label
    :param prediction_label: Numpy array of the prediction label
    :param baseline_prediction_label: Numpy array of the baseline prediction label or None
    :param image: Numpy array of the image
    :param image_id: The full id of the image, used for saving image
    :param show_plot: If true, shows the plot using plt.show()
    :param save_plot_dir: If not None saves the plotted image to this directory
    :return:
    """
    num_plots = 3
    if baseline_prediction_label is not None:
        num_plots += 1

    # Show images, get rid of axis labels
    _, ax = plt.subplots(num_plots)
    for axis in ax:
        axis.imshow(image)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # Draw modal and amodal bounding boxes (same number of labels)
    for idx in range(len(amodal_label)):
        amodal_annotation = amodal_label[idx]
        (aleft, atop, anew_width, anew_height) = rel_to_absolute_label(amodal_annotation)
        arect = patches.Rectangle((aleft, atop), anew_width, anew_height, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(arect)
        # ax[0].text(aleft+PLT_TEXT_OFFSET, atop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[0].set_title('Amodal Bounding Boxes')
        
        modal_annotation = modal_label[idx]
        (mleft, mtop, mnew_width, mnew_height) = rel_to_absolute_label(modal_annotation)
        mrect = patches.Rectangle((mleft, mtop), mnew_width, mnew_height, linewidth=1, edgecolor='b', facecolor='none')
        ax[1].add_patch(mrect)
        # ax[1].text(mleft+PLT_TEXT_OFFSET, mtop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[1].set_title('Modal Bounding Boxes')

    # Draw the model's predicted bounding boxes
    for idx in range(len(prediction_label)):
        prediction_annotation = prediction_label[idx]
        (pleft, ptop, pnew_width, pnew_height) = rel_to_absolute_label(prediction_annotation)
        prect = patches.Rectangle((pleft, ptop), pnew_width, pnew_height, linewidth=1, edgecolor='g', facecolor='none')
        ax[2].add_patch(prect)
        # ax[2].text(pleft+PLT_TEXT_OFFSET, ptop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[2].set_title('Model Predicted Bounding Boxes')

    # Draw the baseline model's predicted bounding boxes
    if baseline_prediction_label is not None:
        for idx in range(len(baseline_prediction_label)):
            baseline_prediction_annotation = baseline_prediction_label[idx]
            (bleft, btop, bnew_width, bnew_height) = rel_to_absolute_label(baseline_prediction_annotation)
            brect = patches.Rectangle((bleft, btop), bnew_width, bnew_height, linewidth=1, edgecolor='g', facecolor='none')
            ax[3].add_patch(brect)
            # ax[3].text(bleft+PLT_TEXT_OFFSET, btop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
            ax[3].set_title('Baseline Predicted Bounding Boxes')

    if show_plot:
        plt.show()
    if save_plot_dir is not None:
        save_filepath = Path(save_plot_dir, image_id + ".png")
        plt.savefig(save_filepath.absolute())
        plt.close()

def plot_bbox_yolo_modal_amodal_predictions_filepaths(amodal_label_filepath: Path, modal_label_filepath: Path, prediction_label_filepath: Path, baseline_prediction_label_filepath: Path, image_filepath: Path, show_plot: bool, save_plot_dir: str):
    """
    Probably don't need this. Can just use plot_bbox_yolo_modal_amodal_predictions_arrays() and do the reading yourself before.
    Plots the bounding boxes for amodal original labels, modal original labels, the model of interest's predicted labels and, optionally, the baseline model's predicted labels.

    :param amodal_label_filepath: Filepath of the amodal label
    :param modal_label_filepath: Filepath of the modal label
    :param prediction_label_filepath: Filepath of the prediction label
    :param baseline_prediction_label_filepath: Filepath of the baseline prediction label
    :param image_filepath: Filepath of the image
    :param show_plot: If true, shows the plot using plt.show()
    :param save_plot_dir: If not None saves the plotted image to this directory
    :return:
    """
    image = np.array(plt.imread(image_filepath))
    image_id = amodal_label_filepath.stem
    amodal_label = read_background_annotation_yolo(amodal_label_filepath)
    modal_label = read_background_annotation_yolo(modal_label_filepath)
    prediction_label = read_background_annotation_yolo(prediction_label_filepath)   # Note: This will have an extra confidence column
    
    baseline_prediction_label = None
    if baseline_prediction_label_filepath is not None:
        baseline_prediction_label = read_background_annotation_yolo(baseline_prediction_label_filepath) # Note: This will have an extra confidence column
    
    plot_bbox_yolo_modal_amodal_predictions_arrays(amodal_label, modal_label, prediction_label, baseline_prediction_label, image, image_id, show_plot, save_plot_dir)
    
def plot_masks_with_image(image: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, label1: np.ndarray, label2: np.ndarray):
    """
    Takes one image, two sets of labels from that image (e.g. predicted and actual, or modal and amodal), the masks of the bboxes in those labels and plots them.

    :param image: Image as numpy array
    :param mask1: Mask of bboxes in label1
    :param mask2: Mask of bboxes in label2
    :param label1: Label 1 of interest for image
    :param label2: Label 2 of interest for image
    :return:
    """
    _, ax = plt.subplots(2,2)

    ax[0][0].imshow(mask1)
    ax[0][0].set_title('Mask 1')
    ax[1][0].imshow(mask2)
    ax[1][0].set_title('Mask 2')

    ax[0][1].imshow(image)
    ax[1][1].imshow(image)
    for idx in range(len(label1)):
        annotation = label1[idx]
        (left, top, new_width, new_height) = rel_to_absolute_label(annotation)
        rect = patches.Rectangle((left, top), new_width, new_height, linewidth=1, edgecolor='r', facecolor='none')
        ax[0][1].add_patch(rect)
        ax[0][1].set_title('Bounding Boxes 1')
    
    for idx in range(len(label2)):
        annotation = label2[idx]
        (left, top, new_width, new_height) = rel_to_absolute_label(annotation)
        rect = patches.Rectangle((left, top), new_width, new_height, linewidth=1, edgecolor='b', facecolor='none')
        ax[1][1].add_patch(rect)
        ax[1][1].set_title('Bounding Boxes 2')

    plt.show()
##########################################################################################
# End Plotting
##########################################################################################


##########################################################################################
# IoU
##########################################################################################
def get_covered_pixels(mask: np.ndarray) -> np.ndarray:
    width, height = mask.shape
    mask_pixel_indices = np.arange(width*height)
    mask_covered_pixels = mask_pixel_indices[mask.flatten().astype(bool)]
    return mask_covered_pixels

def compute_segmentaion_IoU(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_covered_pixels = get_covered_pixels(mask1)
    mask2_covered_pixels = get_covered_pixels(mask2)
    intersecting_pixels = np.intersect1d(mask1_covered_pixels, mask2_covered_pixels)
    union_pixels = np.union1d(mask1_covered_pixels, mask2_covered_pixels)
    IoU = intersecting_pixels.shape[0] / union_pixels.shape[0]
    return IoU

def generate_mask(label_filepath: Path) -> np.ndarray:
    backgroundSize = (IMG_WIDTH, IMG_HEIGHT)
    (backgroundWidth, backgroundHeight) = backgroundSize

    mask = np.ones((backgroundHeight, backgroundWidth))
    label = read_background_annotation_yolo(label_filepath)
    label = clean_yolo(label)
    (N, _) = label.shape

    boundingBoxes = [] 
    for i in range(N):
        boundingBox = rel_to_box_coords(label[i])
        boundingBoxes.append(boundingBox)

    for boundingBox in boundingBoxes:
        (xTopLeft, yTopLeft, xBottomRight, yBottomRight) = boundingBox
        mask[yTopLeft:yBottomRight+1, xTopLeft:xBottomRight+1] = 0

    return np.invert(np.array(mask).astype(bool))

def get_mean_IoU(IoU_and_occlusion_severity_per_image: Dict[str, Tuple[float, float]]) -> float:
    return np.mean([IoU_and_occlusion_severity_per_image[key][0] for key in IoU_and_occlusion_severity_per_image.keys()])

def get_mean_occlusion_severity(IoU_and_occlusion_severity_per_image: Dict[str, Tuple[float, float]]) -> float:
    return np.mean([IoU_and_occlusion_severity_per_image[key][1] for key in IoU_and_occlusion_severity_per_image.keys()])
##########################################################################################
# End IoU
##########################################################################################

##########################################################################################
# Pixelwise Occlusion Severity
##########################################################################################
def compute_pixelwise_occlusion_severity(amodal_mask: np.ndarray, modal_mask: np.ndarray) -> float:
    num_amodal_covered = get_covered_pixels(amodal_mask).shape[0]
    num_modal_covered = get_covered_pixels(modal_mask).shape[0]

    if num_amodal_covered == 0:
        return 0

    occlusion_severity = 1.-num_modal_covered/num_amodal_covered
    return occlusion_severity
##########################################################################################
# End Pixelwise Occlusion Severity
##########################################################################################

##########################################################################################
# Write to file
##########################################################################################
def append_IoU_and_occlusion_severity_to_csv(IoU_and_occlusion_severity_per_image: Dict[str, Tuple[float, float]], filepath: Path, split_name: str):
    logger = logging.getLogger(__name__)
    logger.debug(f"Filepath: {filepath}")
    logger.debug(f"Split name: {split_name}")
    logger.debug(f"IoU_and_occlusion_severity_per_image: {IoU_and_occlusion_severity_per_image}")
    mean_IoU = get_mean_IoU(IoU_and_occlusion_severity_per_image)
    mean_occlusion_severity = get_mean_occlusion_severity(IoU_and_occlusion_severity_per_image)
    
    contents = f"{split_name}, {mean_IoU}, {mean_occlusion_severity}\n"
    with open(filepath, 'a') as file:
        file.write(contents)
##########################################################################################
# End to file
##########################################################################################


def traverse_predicted(yolo_path: str, model_exp_dir: str, baseline_exp_dir: str, model_name: str, baseline_name: str, data_path: str, split_name: str, show_plot: bool, analysis_dir: str, num_examples_to_visualise: int):
    """
    Traverses the file structure of the predictions of your model generated by running test.py from the yolov5 repo.
    The predictions will be under 'yolov5/runs/test/exp<model_exp_num>/labels/'.

    :param yolo_path: path to top level yolov5 directory
    :param model_exp_num: experiment number for model of interest
    :param baseline_exp_num: experiment number for baseline model
    :param data_path: path to model input data (this will be something like datasets-fixed/final on the VM)
    :param split_name: split name you want to examine (e.g. val-side-affixer-different-class)
    :param show_plot: show the plot in matplotlib
    :param save_plot_dir: dir to save plot
    :return:
    """


    severity_vs_goodness_experimental = SeverityVsGoodness(BoxplotHistogramCollector(0., 1.))

    if baseline_exp_dir is not None:
        severity_vs_goodness_baseline = SeverityVsGoodness(BoxplotHistogramCollector(0., 1.))

    IoU_and_occlusion_severity_per_image = {}
    prediction_labels_dir = Path(model_exp_dir, "labels")
    num_labels = len(os.listdir(prediction_labels_dir))
    for prediction_label_filepath in tqdm(os.scandir(prediction_labels_dir)):
        image_id = '.'.join(prediction_label_filepath.name.split('.')[:-1])

        # Show one example
        # if image_id != "000425.17000663":
        #     continue

        image_filepath = Path(data_path, "images", split_name, image_id + ".png")
        amodal_label_filepath = Path(data_path, "labels", split_name, "amodal", prediction_label_filepath.name)
        modal_label_filepath = Path(data_path, "labels", split_name, "modal", prediction_label_filepath.name)
        baseline_prediction_label_filepath = Path(yolo_path, "runs", "test", baseline_exp_dir, "labels", prediction_label_filepath.name) if baseline_exp_dir is not None else None
        
        ##########################################
        # Read image and labels
        ##########################################
        image = np.array(plt.imread(image_filepath))
        image_id = amodal_label_filepath.stem
        amodal_label = read_background_annotation_yolo(amodal_label_filepath)
        modal_label = read_background_annotation_yolo(modal_label_filepath)
        prediction_label = read_background_annotation_yolo(prediction_label_filepath)
        baseline_prediction_label = None
        if baseline_prediction_label_filepath is not None:
            baseline_prediction_label = read_background_annotation_yolo(baseline_prediction_label_filepath)

        ##########################################
        # PUT THE ANALYSIS FUNCTIONS YOU WANT HERE
        ##########################################
        if baseline_prediction_label_filepath is not None:
            severity_vs_goodness_baseline.process_image(amodal_label, modal_label, baseline_prediction_label)
        severity_vs_goodness_experimental.process_image(amodal_label, modal_label, prediction_label)

        # IoU
        amodal_mask = generate_mask(amodal_label_filepath)
        modal_mask = generate_mask(modal_label_filepath)
        prediction_mask = generate_mask(prediction_label_filepath)
        # plot_masks_with_image(image, amodal_mask, modal_mask, amodal_label, modal_label)
        IoU = compute_segmentaion_IoU(amodal_mask, prediction_mask)
        pixelwise_occlusion_severity = compute_pixelwise_occlusion_severity(amodal_mask, modal_mask)
        IoU_and_occlusion_severity_per_image[image_id] = (IoU, pixelwise_occlusion_severity)

        # TODO: this seems expensive, can we randomly subsample this or only do it in interesting cases?
        # Plotting
        if random.random() < num_examples_to_visualise/num_labels:
            plot_bbox_yolo_modal_amodal_predictions_arrays(amodal_label, modal_label, prediction_label, baseline_prediction_label, image, image_id, show_plot, Path(analysis_dir, split_name, f"{model_name}_predictions"))

    if baseline_exp_dir is not None:
        """
        Phil, I left this here because I didn't want to add lots of args to your script in your stead. Do with this what you will.
        """
        # TODO: allow configuration of num_bins from argparse?
        # TODO: make savepath configurable from argparse
        severity_vs_goodness_baseline.collector.produce_histogram(
            num_bins=10, title="Baseline Model - Prediction IoU by Occlusion Severity",
            x_label="Occlusion Severity", y_label="predicted box/ground truth IoU", savepath=f"{analysis_dir}/{split_name}/histograms/{baseline_name}.png"
        )
        # TODO: make title configurable from argparse to generate more specific title here
    severity_vs_goodness_experimental.collector.produce_histogram(
        num_bins=10, title=f"{model_name} - Prediction IoU by Occlusion Severity",
        x_label="Occlusion Severity", y_label="predicted box/ground truth IoU", savepath=f"{analysis_dir}/{split_name}/histograms/{model_name}.png"
    )
    append_IoU_and_occlusion_severity_to_csv(IoU_and_occlusion_severity_per_image, f"{analysis_dir}/iou_and_occlusion_severity.csv", split_name)


if __name__ == "__main__":
    """
    Example Usage:
    python phil/analyse.py --data /Users/philipmateopfeffer/Downloads/cs231n_class_project/fixed_final \
        --yolo /Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5 \
        --model-exp-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5/runs/test/exp41/ \
        --baseline-exp-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5/runs/test/exp41/ \
        --baseline-name baseline \
        --analysis-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/analysis/ \
        --model-name baseline-amodal-combined-train \
        --split-name val-side-affixer-different-class \
        --num-examples-to-visualise 0 \
        --show-plot

    Prereqs:
        - Final data of the form {path}/final/{split_name}, e.g. /datasets-fixed/final/val-side-affixer-different-class
        - Results of running yolov5/test.py on the final data. This will give you the model_exp_num. Here is an example:
            `python test.py --img 640 --batch-size 1 --data ../fixed_final.yaml --weights ../baseline-amodal-combined-train/weights/best.pt --task val --save-hybrid --save-conf --conf-thres 0.25 --iou-thres 0.6`
    """
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-exp-dir', type=str, default=None, help='experiment output dir for model of interest')
    parser.add_argument('--baseline-exp-dir', type=str, default=None, help='experiment output dir for baseline model')
    parser.add_argument('--yolo', type=str, default='./yolov5', help='path to top level yolov5 directory')
    parser.add_argument('--data', type=str, default='./data', help='path to model input data')
    parser.add_argument('--split-name', type=str, default='test', help='split name')
    parser.add_argument('--model-name', type=str, default=None, help='model name')
    parser.add_argument('--baseline-name', type=str, default=None, help='baseline model name')
    parser.add_argument('--show-plot', action='store_true', help='show the plot in matplotlib')
    parser.add_argument('--analysis-dir', type=str, default=None, help='dir to save plot')
    parser.add_argument('--num-examples-to-visualise', type=int, default=5, help='num examples to visualise (non-deterministic)')
    args = parser.parse_args()

    traverse_predicted(args.yolo, args.model_exp_dir, args.baseline_exp_dir, args.model_name, args.baseline_name, args.data, args.split_name, args.show_plot, args.analysis_dir, args.num_examples_to_visualise)
    logger.info('Finished')


"""
---------
GRAVEYARD
---------
def traverse_datasets_final(exp_num: int, yolo_path: str, data_path: str, show_plot: bool, save_plot_dir: str):
    for images_or_labels in os.scandir(data_path):
        if images_or_labels.name.startswith('.DS_Store'):
            continue
    
        if images_or_labels.name == "labels":
            for data_split in os.scandir(images_or_labels):
                if data_split.name.startswith('.DS_Store') or data_split.name.endswith('.cache') or data_split.name == "histograms":
                    continue

                modal_dir = Path(data_split.path, "modal")
                amodal_dir = Path(data_split.path, "amodal")
                for modal_label_filepath in os.scandir(modal_dir):
                    if modal_label_filepath.name.startswith('.DS_Store'):
                        continue

                    modal_label_filepath_list = modal_label_filepath.name.split('.')
                    original_image_id = modal_label_filepath_list[0]
                    amodal_label_filepath = Path(amodal_dir, modal_label_filepath.name)
                    image_filepath = Path(data_path, "images", data_split.name, original_image_id + ".png")
                    prediction_label_filepath = Path(yolo_path, "runs", data_split.name, "exp" + str(exp_num), "labels", modal_label_filepath.name)
                    plot_bbox_yolo_modal_amodal_predictions(amodal_label_filepath, modal_label_filepath, prediction_label_filepath, image_filepath, show_plot, save_plot_dir)
"""
