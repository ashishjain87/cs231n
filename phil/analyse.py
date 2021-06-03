import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

IMG_WIDTH = 1224
IMG_HEIGHT = 370
PLT_TEXT_OFFSET = -10

def rel_to_absolute_label(annotation):
    x, y, width, height = annotation[1:5]
    new_width = math.floor(width * IMG_WIDTH)
    new_height = math.floor(height * IMG_HEIGHT)
    left = math.floor(x * IMG_WIDTH) - new_width//2
    top = math.floor(y * IMG_HEIGHT) - new_height//2
    return (left, top, new_width, new_height)

def rel_to_box_coords(annotation):
    x, y, width, height = annotation[1:5]
    new_width = math.floor(width * IMG_WIDTH)
    new_height = math.floor(height * IMG_HEIGHT)
    left = math.floor(x * IMG_WIDTH) - new_width//2
    top = math.floor(y * IMG_HEIGHT) - new_height//2
    right = left + new_width
    bottom = top + new_height
    return (left, top, right, bottom)

def clean_yolo(yolo_label):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def read_background_annotation_yolo(yolo_label_path: str) -> np.ndarray:
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 

def plot_bbox_yolo_modal_amodal_predictions(amodal_label_filepath: Path, modal_label_filepath: Path, prediction_label_filepath: Path, baseline_prediction_label_filepath: Path, image_filepath: Path, show_plot: bool, save_plot_dir: str):
    image = np.array(plt.imread(image_filepath))
    amodal_label = read_background_annotation_yolo(amodal_label_filepath)
    modal_label = read_background_annotation_yolo(modal_label_filepath)
    prediction_label = read_background_annotation_yolo(prediction_label_filepath)
    num_plots = 3

    baseline_prediction_label = None
    if baseline_prediction_label_filepath is not None:
        num_plots += 1
        baseline_prediction_label = read_background_annotation_yolo(baseline_prediction_label_filepath)
    
    _, ax = plt.subplots(num_plots)
    for axis in ax:
        axis.imshow(image)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # Draw each bounding box
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

    for idx in range(len(prediction_label)):
        prediction_annotation = prediction_label[idx]
        (pleft, ptop, pnew_width, pnew_height) = rel_to_absolute_label(prediction_annotation)
        prect = patches.Rectangle((pleft, ptop), pnew_width, pnew_height, linewidth=1, edgecolor='g', facecolor='none')
        ax[2].add_patch(prect)
        # ax[2].text(pleft+PLT_TEXT_OFFSET, ptop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[2].set_title('Model Predicted Bounding Boxes')

    if baseline_prediction_label_filepath is not None:
        for idx in range(len(baseline_prediction_label)):
            baseline_prediction_annotation = baseline_prediction_label[idx]
            (bleft, btop, bnew_width, bnew_height) = rel_to_absolute_label(baseline_prediction_annotation)
            brect = patches.Rectangle((bleft, btop), bnew_width, bnew_height, linewidth=1, edgecolor='g', facecolor='none')
            ax[3].add_patch(brect)
            # ax[2].text(pleft+PLT_TEXT_OFFSET, ptop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
            ax[3].set_title('Baseline Predicted Bounding Boxes')

    if show_plot:
        plt.show()
    if save_plot_dir is not None:
        save_filepath = Path(save_plot_dir, amodal_label_filepath.stem + ".png")
        plt.savefig(save_filepath.absolute())
        plt.close()

def plot_masks_with_image(mask1: np.ndarray, mask2: np.ndarray, image_filepath: Path, label1_filepath: Path, label2_filepath: Path):
    _, ax = plt.subplots(2,2)
    image = np.array(plt.imread(image_filepath))
    label1 = read_background_annotation_yolo(label1_filepath)
    label2 = read_background_annotation_yolo(label2_filepath)

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

def compute_segmentaion_IoU(mask1: np.ndarray, mask2: np.ndarray) -> float:
    return 0

def get_split_type(split: str) -> str:
    if split.startswith("train"): return "train"
    if split.startswith("val"): return "val"
    if split.startswith("test"): return "test"
    assert False, "Split type not recognised."

def traverse_predicted(yolo_path: str, model_exp_num: int, baseline_exp_num: int, data_path: str, split_name: str, show_plot: bool, save_plot_dir: str):
    prediction_labels_dir = Path(yolo_path, "runs", "test", "exp" + str(model_exp_num), "labels")
    for prediction_label_filepath in os.scandir(prediction_labels_dir):
        image_id = '.'.join(prediction_label_filepath.name.split('.')[:-1])

        # Show one example
        # if image_id.find("000103.17000472") == -1:
        #     continue

        image_filepath = Path(data_path, "images", split_name, image_id + ".png")
        amodal_label_filepath = Path(data_path, "labels", split_name, "amodal", prediction_label_filepath.name)
        modal_label_filepath = Path(data_path, "labels", split_name, "modal", prediction_label_filepath.name)
        baseline_prediction_label_filepath = Path(yolo_path, "runs", "test", "exp" + str(baseline_exp_num), "labels", prediction_label_filepath.name) if baseline_exp_num is not None else None
        
        ##########################################
        # PUT THE ANALYSIS FUNCTIONS YOU WANT HERE
        ##########################################
        # IoU
        amodal_mask = generate_mask(amodal_label_filepath)
        modal_mask = generate_mask(modal_label_filepath)
        plot_masks_with_image(amodal_mask, modal_mask, image_filepath, amodal_label_filepath, modal_label_filepath)
        IoU = compute_segmentaion_IoU(amodal_mask, modal_mask)

        # Plotting
        plot_bbox_yolo_modal_amodal_predictions(amodal_label_filepath, modal_label_filepath, prediction_label_filepath.path, baseline_prediction_label_filepath, image_filepath, show_plot, save_plot_dir)


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

        return np.array(mask)


if __name__ == "__main__":
    """
    Example Usage:
    python analyse.py --data /Users/philipmateopfeffer/Downloads/cs231n_class_project/fixed_final \
        --yolo /Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5 --model-exp-num 41 --baseline-exp-num 41 \
        --save-plot-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/analysis/modal-amodal-prediction-plots \
        --split-name val-side-affixer-different-class \
        --show-plot
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-exp-num', type=int, default=0, help='experiment number for model of interest')
    parser.add_argument('--baseline-exp-num', type=int, default=None, help='experiment number for baseline model')
    parser.add_argument('--yolo', type=str, default='./yolov5', help='path to top level yolov5 directory')
    parser.add_argument('--data', type=str, default='./data', help='path to model input data')
    parser.add_argument('--split-name', type=str, default='test', help='split name')
    parser.add_argument('--show-plot', action='store_true', help='show the plot in matplotlib')
    parser.add_argument('--save-plot-dir', type=str, default=None, help='dir to save plot')
    args = parser.parse_args()

    if args.save_plot_dir is not None and not os.path.exists(args.save_plot_dir):
        os.makedirs(args.save_plot_dir)

    traverse_predicted(args.yolo, args.model_exp_num, args.baseline_exp_num, args.data, args.split_name, args.show_plot, args.save_plot_dir)




"""
---------
GRAVEYARD
---------
def traverse_original(exp_num: int, yolo_path: str, data_path: str, show_plot: bool, save_plot_dir: str):
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