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

def clean_yolo(yolo_label):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def read_background_annotation_yolo(yolo_label_path: str) -> np.ndarray:
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 


def plot_bbox_yolo_modal_amodal_predictions(amodal_label_filepath: Path, modal_label_filepath: Path, prediction_label_filepath: Path, image_filepath: Path, show_plot: bool, save_plot_dir: str):
    image = np.array(plt.imread(image_filepath))
    amodal_label = read_background_annotation_yolo(amodal_label_filepath)
    modal_label = read_background_annotation_yolo(modal_label_filepath)
    prediction_label = read_background_annotation_yolo(prediction_label_filepath)
    
    _, ax = plt.subplots(3)
    ax[0].imshow(image)
    ax[1].imshow(image)
    ax[2].imshow(image)

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

        prediction_annotation = prediction_label[idx]
        (pleft, ptop, pnew_width, pnew_height) = rel_to_absolute_label(prediction_annotation)
        prect = patches.Rectangle((pleft, ptop), pnew_width, pnew_height, linewidth=1, edgecolor='b', facecolor='none')
        ax[2].add_patch(prect)
        # ax[2].text(pleft+PLT_TEXT_OFFSET, ptop+PLT_TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[2].set_title('Predicted Bounding Boxes')

    if show_plot:
        plt.show()
    if save_plot_dir is not None:
        save_filepath = Path(amodal_label_filepath.stem)
        print(save_filepath)
        # plt.savefig(save_plot_dir)

def traverse_original(exp_num: int, yolo_path: str, data_path: str, show_plot: bool, save_plot_dir: str):
    for images_or_labels in os.scandir(data_path):
        if images_or_labels.name.startswith('.DS_Store'):
            continue
        # print(images_or_labels.name)
    
        if images_or_labels.name == "labels":
            # Labels 
            for data_split in os.scandir(images_or_labels):
                if data_split.name.startswith('.DS_Store') or data_split.name.endswith('.cache') or data_split.name == "histograms":
                    continue
                # print(f"\t{data_split.name}")

                # print(f"\t\tmodal")
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

                    # print(f"\t\t\tModal:  {modal_label_filepath.path}")
                    # print(f"\t\t\tAmodal: {amodal_label_filepath}")
                    # print(f"\t\t\tImage:  {image_filepath}")
                    # print("\t\t\t------------------------------------------")

                    plot_bbox_yolo_modal_amodal_predictions(amodal_label_filepath, modal_label_filepath, prediction_label_filepath, image_filepath, show_plot, save_plot_dir)

def get_split_type(split: str) -> str:
    if split.startswith("train"): return "train"
    if split.startswith("val"): return "val"
    if split.startswith("test"): return "test"
    assert False, "Split type not recognised."

def traverse_predicted(exp_num: int, yolo_path: str, data_path: str, split_name: str, show_plot: bool, save_plot_dir: str):
    # split_type = get_split_type(split_name)
    prediction_labels_dir = Path(yolo_path, "runs", "test", "exp" + str(exp_num), "labels")
    for prediction_label_filepath in os.scandir(prediction_labels_dir):
        prediction_label_filepath_list = prediction_label_filepath.name.split('.')
        original_image_id = prediction_label_filepath_list[0]
        if original_image_id != "004948":
            continue
        augmented_image_id = "." + prediction_label_filepath_list[1] if len(prediction_label_filepath_list) == 3 else ""

        image_filepath = Path(data_path, "images", split_name, original_image_id + augmented_image_id + ".png")
        amodal_label_filepath = Path(data_path, "labels", split_name, "amodal", prediction_label_filepath.name)
        modal_label_filepath = Path(data_path, "labels", split_name, "modal", prediction_label_filepath.name)

        print(image_filepath)
        print(prediction_label_filepath)
        print(amodal_label_filepath)
        print(modal_label_filepath)
        plot_bbox_yolo_modal_amodal_predictions(amodal_label_filepath, modal_label_filepath, prediction_label_filepath.path, image_filepath, show_plot, save_plot_dir)




if __name__ == "__main__":
    """
    Example Usage:
    python analyse.py --data /Users/philipmateopfeffer/Downloads/cs231n_class_project/final \
        --yolo /Users/philipmateopfeffer/Downloads/cs231n_class_project/yolov5 --exp-num 35 \
        --save-plot-dir /Users/philipmateopfeffer/Downloads/cs231n_class_project/analysis/modal-amodal-prediction-plots \
        --split-name val-side-affixer-different-class \
        --show-plot
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-num', type=int, default=0, help='experiment number')
    parser.add_argument('--yolo', type=str, default='./yolov5', help='path to top level yolov5 directory')
    parser.add_argument('--data', type=str, default='./data', help='path to model input data')
    parser.add_argument('--split-name', type=str, default='test', help='split name')
    parser.add_argument('--show-plot', action='store_true', help='show the plot in matplotlib')
    parser.add_argument('--save-plot-dir', type=str, default=None, help='dir to save plot')
    args = parser.parse_args()

    if args.save_plot_dir is not None and not os.path.exists(args.save_plot_dir):
        os.makedirs(args.save_plot_dir)

    # traverse_original_and_predicted(args.exp_num, args.yolo, args.data, args.show_plot, args.save_plot_dir)
    traverse_predicted(args.exp_num, args.yolo, args.data, args.split_name, args.show_plot, args.save_plot_dir)

"""
TODO: Need to be able to take in the baseline to compare
"""