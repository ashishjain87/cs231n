import os
from pathlib import Path

def get_immediate_subdirectories(super_dir):
    return [name for name in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, name))]

def get_immediate_subdirectory_paths(super_dir):
    subdirs = get_immediate_subdirectories(super_dir)
    return [os.path.join(super_dir, subdir) for subdir in subdirs]

def get_background_annotation_file_path(image_file_path, annotations_dir_path):
    # Add support for dealing with scenario where images and annotations are in the same directory
    common_path = get_common_path(image_file_path, annotations_dir_path)
    annotation_folder_name = get_leaf_folder(annotations_dir_path) 
    image_file_name_without_ext = get_filename_without_ext(image_file_path)
    background_annotations_file_name = '%s.%s' % (image_file_name_without_ext, 'txt') 
    return os.path.join(common_path, annotation_folder_name, background_annotations_file_name)

def get_common_path(path1, path2):
    return os.path.commonpath([path1, path2])

def get_immediate_grandparent_folder(abs_path):
    path = Path(path)
    # TODO: Add support for checking whether grandparent directory actually exists before returning.
    return str(path.parents[1])

def get_immediate_parent_folder(abs_path):
    folders = get_all_folders_in_path(abs_path)
    return folders[0] # assuming path is not at root

def get_filename_without_ext(path):
    filename_with_ext = os.path.basename(path)
    list = os.path.splitext(filename_with_ext)
    return list[0]

def get_leaf_folder(path):
    if os.path.isdir(path):
        return Path(path).name
    else:
        folders = get_all_folders_in_path(path)
        return folders[0]

def get_filename_and_extension(path): 
    path = Path(path)
    return (path.stem, path.suffix)

def get_all_folders_in_path(path): 
    drive, path_and_file = os.path.splitdrive(path)
    path, file = os.path.split(path_and_file)

    folders = []
    while True:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    return folders
