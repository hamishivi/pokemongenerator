"""
Util script moving the segmentation dataset I got (http://groups.csail.mit.edu/vision/datasets/ADE20K/)
into appropriate folders, ordered correctly.
This was important for making keras training easier.
"""

import os

directory = 'segmentation_dataset/images/training'


def get_all_files_in_folder(folder_path):
    for file in os.listdir(folder_path):
        filename = os.path.join(folder_path, file)
        if os.path.isdir(filename) or 'DS_Store' in filename:
            continue
        elif '_seg' in filename:
            os.rename(filename, "segmentation_dataset/images/masks/" + file)
            print("segmentation_dataset/images/masks/" + file)
        else:
            os.rename(filename, "segmentation_dataset/images/raws/" + file)
            print("segmentation_dataset/images/raws/" + file)
for folder in os.listdir(directory):
    filename = os.path.join(directory, folder)
    if 'DS_Store' in filename:
        continue
    elif os.path.isdir(filename):
        for sub_folder in os.listdir(filename):
            if not os.path.isdir(os.path.join(filename, sub_folder)):
                get_all_files_in_folder(filename)
                break
            else:
                get_all_files_in_folder(os.path.join(filename, sub_folder))
    else:
        get_all_files_in_folder(filename)