import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from tqdm import tqdm
import argparse

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


def copy_label_file(path_to_top_level_input_dir: str, path_to_containing_dir: str, out_dir_set_name: str,
                    out_dir_top_level_path: str, original_label_file_name: str, output_label_file_name: str):
    if not path_to_containing_dir.startswith(path_to_top_level_input_dir):
        raise ValueError("The path computation is erroneous.")
    path_suffix = path_to_containing_dir[len(path_to_top_level_input_dir + "/images/"):]
    os.system(
        f"cp {path_to_top_level_input_dir}/labels/{path_suffix}/modal/{original_label_file_name} {out_dir_top_level_path}/labels/{out_dir_set_name}/modal/{output_label_file_name}"
    )
    os.system(
        f"cp {path_to_top_level_input_dir}/labels/{path_suffix}/amodal/{original_label_file_name} {out_dir_top_level_path}/labels/{out_dir_set_name}/amodal/{output_label_file_name}"
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
    os.system(f"mkdir {out_dir_top_level_path}/labels/{out_dir_set_name}/modal")
    os.system(f"mkdir {out_dir_top_level_path}/labels/{out_dir_set_name}/amodal")

    print(f"Populating output directories for {out_dir_set_name} set...")
    for image_id in tqdm(chosen_images.keys()):
        images_to_use = chosen_images[image_id]
        original_label_file_name = f"{image_id}.txt"
        for image in images_to_use:
            path_to_containing_dir, filename = image
            extension_len = len(FILE_EXTENSION)
            output_label_file_name = filename[:(-1*extension_len)] + "txt"
            copy_label_file(path_to_top_level_input_dir, path_to_containing_dir,
                            out_dir_set_name, out_dir_top_level_path,
                            original_label_file_name, output_label_file_name)
            copy_image(f"{path_to_containing_dir}/{filename}", filename, out_dir_set_name, out_dir_top_level_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-root', type=str, help="top-level dir where input data is stored")
    parser.add_argument('--out-root', type=str, help="top-level dir where output data should be stored")
    parser.add_argument('--use-aug-prob', type=float, default=0.4,
                        help="a fraction between 0 and 1 dictating how often to use augmented images.")
    parser.add_argument('--clear-out-dirs', action='store_true', help="remove files in output directories before generation")
    parser.add_argument('--skip-train', action='store_true', help="skip processing training data")
    parser.add_argument('--skip-val-test', action='store_true', help="skip processing val and test data")

    return parser.parse_args()


def make_dir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.system(f"mkdir {dirpath}")


def rm_contents_if_exists(dirpath: str):
    if os.path.exists(dirpath):
        os.system(f"rm -r {dirpath}")


def collect_images_into_final_sets(path_to_top_level_input_dir: str,
                                   path_to_top_level_output_dir: str,
                                   use_aug_prob: float,
                                   clear_out_dirs: bool,
                                   skip_train: bool,
                                   skip_val_test: bool):
    # As a user of this, consider setting a random seed
    # random.seed(42)

    make_dir_if_not_exists(path_to_top_level_output_dir)
    make_dir_if_not_exists(f"{path_to_top_level_output_dir}/images")
    make_dir_if_not_exists(f"{path_to_top_level_output_dir}/labels")

    for setname in os.listdir(f"{path_to_top_level_input_dir}/images"):
        if setname.startswith("train"):
            if skip_train:
                print(f"skipping {setname}")
                continue
            print(f"processing {setname}")
            images = get_chosen_images(f"{path_to_top_level_input_dir}/images/{setname}", False, use_aug_prob)
            if clear_out_dirs:
                rm_contents_if_exists(f"{path_to_top_level_output_dir}/images/{setname}")
                rm_contents_if_exists(f"{path_to_top_level_output_dir}/labels/{setname}")
            populate_out_dir(images, path_to_top_level_input_dir, path_to_top_level_output_dir, setname)
        elif setname.startswith("val") or setname.startswith("test"):
            if skip_val_test:
                print(f"skipping {setname}")
                continue
            print(f"processing {setname}")
            images = get_chosen_images(f"{path_to_top_level_input_dir}/images/{setname}", True, use_aug_prob)
            if clear_out_dirs:
                rm_contents_if_exists(f"{path_to_top_level_output_dir}/images/{setname}")
                rm_contents_if_exists(f"{path_to_top_level_output_dir}/labels/{setname}")
            populate_out_dir(images, path_to_top_level_input_dir, path_to_top_level_output_dir, setname)
        else:
            print(f"Could not process '{setname}', must start with 'train', 'val', or 'test'")


if __name__ == "__main__":
    args = get_args()
    in_path = args.in_root
    out_path = args.out_root
    aug_prob = args.use_aug_prob
    clear_out_dirs = args.clear_out_dirs
    skip_train = args.skip_train
    skip_val_test = args.skip_val_test
    collect_images_into_final_sets(in_path, out_path, aug_prob, clear_out_dirs, skip_train, skip_val_test)
