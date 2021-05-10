import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from tqdm import tqdm

FILE_EXTENSION = "png"
AUG_REGEX = "(\d+)\..*\." + FILE_EXTENSION
RAW_REGEX = "(\d+)\." + FILE_EXTENSION


def get_image_mapping(path_to_images: str, regex: str) -> Dict[str, Tuple[str, str]]:
    raw_filenames_pre = os.listdir(path_to_images)
    mapping = {}
    for filename in raw_filenames_pre:
        match = re.match(regex, filename)
        if match is not None:
            image_id = match.group(1)
            mapping[image_id] = (path_to_images, filename)
    return mapping


def get_aug_dirs(path_to_aug_dir: str) -> List[str]:
    aug_dirs = os.listdir(path_to_aug_dir)
    return [f"{path_to_aug_dir}/{aug_dir}" for aug_dir in aug_dirs]


def get_chosen_images(path_to_base_dir: str, one_per_id: bool = False, use_aug_prob: float = 0.4) -> Dict[str, List[Tuple[str, str]]]:
    """

    :param path_to_base_dir: A path to a folder like '.../images/train'
    :param one_per_id: if True, don't use more than one image created from the same image in the KITTI dataset
    :param use_aug_prob: if one_per_id, probability of selecting an augmented image for the image id uniformly at random from those available.
        If not one_per_id, probability of including an individual augmentation.
    :return:
    """

    aug_dirs = get_aug_dirs(f"{path_to_base_dir}/aug")

    orig_mapping = get_image_mapping(f"{path_to_base_dir}/orig", RAW_REGEX)
    aug_mappings = [get_image_mapping(aug_dir, AUG_REGEX) for aug_dir in aug_dirs]

    chosen_images = defaultdict(lambda: list())

    print(f"Choosing images from {path_to_base_dir}")
    for image_id in tqdm(orig_mapping.keys()):
        if one_per_id:
            aug_candidates = []
            for mapping in aug_mappings:
                if image_id in mapping:
                    aug_candidates.append(mapping[image_id])
            if len(aug_candidates) > 0 and random.random() <= use_aug_prob:
                chosen_images[image_id].append(random.choice(aug_candidates))
            else:
                chosen_images[image_id].append(orig_mapping[image_id])
        else:
            chosen_images[image_id].append(orig_mapping[image_id])
            for mapping in aug_mappings:
                if image_id in mapping:
                    if random.random() <= use_aug_prob:
                        chosen_images[image_id].append(mapping[image_id])


    return chosen_images


def copy_label_file(path_to_top_level_input_dir: str, out_dir_set_name: str, out_dir_top_level_path: str,
                    original_label_file_name: str, output_label_file_name: str):
    os.system(
        f"cp {path_to_top_level_input_dir}/labels/{out_dir_set_name}/orig/{original_label_file_name} {out_dir_top_level_path}/labels/{out_dir_set_name}/{output_label_file_name}"
    )


def copy_image(path_to_image: str, image_name: str, out_dir_set_name: str, out_dir_top_level_path: str):
    os.system(
        f"cp {path_to_image} {out_dir_top_level_path}/images/{out_dir_set_name}/{image_name}"
    )


def populate_out_dir(chosen_images: Dict[str, List[Tuple[str, str]]], path_to_top_level_input_dir: str, out_dir_top_level_path: str, out_dir_set_name: str):
    """

    :param chosen_images: output of get_chosen_images
    :param path_to_top_level_input_dir: a path to the top-level *input* directory, such as one containing 'images/train/... and 'labels/val/...'
    :param out_dir_top_level_path: the path to the directory where output files will be saved (containing 'images/train/...', etc.)
    :param out_dir_set_name: the name of the dataset (e.g., "train", "val", etc.)
    :return:
    """

    os.system(f"mkdir {out_dir_top_level_path}/images/{out_dir_set_name}")
    os.system(f"mkdir {out_dir_top_level_path}/labels/{out_dir_set_name}")

    print(f"Populating output directories for {out_dir_set_name} set...")
    for image_id in tqdm(chosen_images.keys()):
        images_to_use = chosen_images[image_id]
        original_label_file_name = f"{image_id}.txt"
        for image in images_to_use:
            path_to_containing_dir, filename = image
            extension_len = len(FILE_EXTENSION)
            output_label_file_name = filename[:(-1*extension_len)] + "txt"
            copy_label_file(path_to_top_level_input_dir, out_dir_set_name, out_dir_top_level_path,
                            original_label_file_name, output_label_file_name)
            copy_image(f"{path_to_containing_dir}/{filename}", filename, out_dir_set_name, out_dir_top_level_path)


def collect_images_into_final_sets(path_to_top_level_input_dir: str, path_to_top_level_output_dir: str):
    # As a user of this, consider setting a random seed
    # random.seed(42)

    os.system(f"mkdir {path_to_top_level_output_dir}")
    os.system(f"mkdir {path_to_top_level_output_dir}/images")
    os.system(f"mkdir {path_to_top_level_output_dir}/labels")

    train_images = get_chosen_images(f"{path_to_top_level_input_dir}/images/train", False, 0.4)
    populate_out_dir(train_images, path_to_top_level_input_dir, path_to_top_level_output_dir, "train")

    for setname in ["val", "test"]:
        train_images = get_chosen_images(f"{path_to_top_level_input_dir}/images/{setname}", True, 0.4)
        populate_out_dir(train_images, path_to_top_level_input_dir, path_to_top_level_output_dir, setname)


if __name__ == "__main__":
    in_path = "/Users/schwartzd/dev/classes/cs231n/project/intermediate_data"
    out_path = "/Users/schwartzd/dev/classes/cs231n/project/processed_data"
    collect_images_into_final_sets(in_path, out_path)