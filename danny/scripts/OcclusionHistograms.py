from typing import Tuple, Optional, List, Set, Dict
from matplotlib import pyplot as plt
import os
import argparse
import numpy as np
from seaborn import heatmap

CLASS_MAP = {0: 'car', 1: 'van', 2: 'truck', 3: 'pedestrian', 4: 'person sitting', 5: 'cyclist', 6: 'tram', 7: 'misc', 8: 'dontcare'}


class YoloLabel:
    category: int
    center_x: float
    center_y: float
    width: float
    height: float

    def __init__(self, category: int, center_x: float, center_y: float, width: float, height: float):
        self.category = category
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height


def intersection(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    a_s, a_e = a
    b_s, b_e = b
    if a_s <= b_e and b_s <= a_e:
        overlap = max(a_s, b_s), min(a_e, b_e)
        if overlap[0] == overlap[1]:
            return None
        else:
            return overlap
    return None


def overlap(a: YoloLabel, b: YoloLabel) -> Tuple[bool, float]:
    """

    :param a: a YoloLabel
    :param b: a different YoloLabel
    :return: a bool indicating whether the two labels overlap and an IoU score if they do overlap
    """
    x_intersection = intersection((a.center_x - a.width/2, a.center_x + a.width/2),
                                  (b.center_x - b.width/2, b.center_x + b.width/2))
    y_intersection = intersection((a.center_y - a.height / 2, a.center_y + a.height / 2),
                                  (b.center_y - b.height / 2, b.center_y + b.height / 2))
    if x_intersection is None or y_intersection is None:
        return False, 0.

    intersect_area = (x_intersection[1] - x_intersection[0])*(y_intersection[1] - y_intersection[0])
    a_area = a.width * a.height
    b_area = b.width * b.height
    iou = intersect_area/(a_area + b_area - intersect_area)

    return True, iou


def load_labels(path_to_labels: str) -> List[List[YoloLabel]]:
    label_filenames = os.listdir(path_to_labels)
    images = []
    for filename in label_filenames:
        if not filename.endswith(".txt"):
            continue
        labels = []
        with open(f"{path_to_labels}/{filename}", "r") as file:
            for line in file:
                parts = line.split()
                category = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                if category != 8:
                    # Ignore 'dontcare' annotations
                    labels.append(YoloLabel(category, center_x, center_y, width, height))
        images.append(labels)
    return images


def profile_dataset(images: List[List[YoloLabel]], hist_out_dir: str, severity_bins: int, heatmap_iou_threshold: float):
    num_occlusions_by_image = []
    occlusions = []  # each entry is of the form (iou, YoloLabel a, YoloLabel b)

    for image in images:
        num_occlusions = 0
        for i in range(len(image)):
            a = image[i]
            for j in range(i+1, len(image)):
                b = image[j]
                occlusion, iou = overlap(a, b)
                if occlusion:
                    occlusions.append((iou, a, b))
                    num_occlusions += 1
        num_occlusions_by_image.append(num_occlusions)

    plt.figure()
    plt.hist(num_occlusions_by_image, bins=(max(num_occlusions_by_image) + 1))
    plt.xlabel("Number of occlusions in the image")
    plt.ylabel("Number of images with this many occlusions")
    plt.title("Occlusion Occurrence Histogram")
    plt.savefig(f"{hist_out_dir}/occurrence.png")

    occlusion_ious = [o[0] for o in occlusions]
    plt.figure()
    plt.hist(occlusion_ious, bins=severity_bins)
    plt.xlabel("Bounding box IoU range")
    plt.ylabel("Number of occlusions in dataset with approximately this IoU")
    plt.title("Occlusion Severity Histogram")
    plt.savefig(f"{hist_out_dir}/severity.png")

    plt.figure()
    create_occlusion_heatmap(occlusions, CLASS_MAP, heatmap_iou_threshold, hist_out_dir)


def create_occlusion_heatmap(occlusions: List[Tuple[float, YoloLabel, YoloLabel]],
                             idx_to_classname: Dict[int, str],
                             iou_threshold: float,
                             hist_out_dir: str):
    num_classes = len(idx_to_classname.keys()) - 1  # prune 'dontcare'
    mtx = np.zeros((num_classes, num_classes))
    num_above_threshold = 0.
    for occlusion in occlusions:
        if occlusion[0] >= iou_threshold:
            num_above_threshold += 1
            class1 = occlusion[1].category
            class2 = occlusion[2].category
            if class1 == class2:
                mtx[class1, class2] += 1
            else:
                mtx[class1, class2] += 1
                mtx[class2, class1] += 1

    if num_above_threshold == 0:
        print(f"IoU threshold ({iou_threshold}) for occlusion heatmap is too large! Skipping heatmap generation...")
        return

    # normalize by number of 'interesting' occlusions
    mtx /= num_above_threshold

    class_names = [idx_to_classname[i] for i in range(num_classes)]

    fig = heatmap(data=mtx, xticklabels=class_names, yticklabels=class_names, annot=True).get_figure()
    fig.tight_layout(pad=3.2)
    plt.title("Heatmap of overlapping objects in dataset\n(as a proportion of total occlusions)")
    plt.xlabel("Object 1 category")
    plt.ylabel("Object 2 category")
    fig.savefig(f"{hist_out_dir}/occlusion_heatmap.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-label-files', type=str, help="path to a directory containing label files")
    parser.add_argument('--hist-out-dir', type=str, help="path to a directory where generated histograms should be saved")
    parser.add_argument('--severity-bins', type=int, default=16, help="number of bins for the occlusion severity histogram")
    parser.add_argument('--heatmap_iou_threshold', type=float, default=0., help="minimum IoU score for an occlusion to be included in the generated heatmap")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    image_labels = load_labels(args.path_to_label_files)
    profile_dataset(image_labels, args.hist_out_dir, args.severity_bins, args.heatmap_iou_threshold)

