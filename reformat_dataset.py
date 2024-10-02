# This code is used to reformat the dataset folder structure from Roboflow format
# to the format required for knowledge distillation datasets.
# Please modify the folder paths accordingly.

import os
import shutil
import warnings
from tqdm import tqdm

# Define the source and destination folders
# Modify the source and destination folder paths based on your dataset.
source_folder = {
    'train_images': 'source_datasets/datasets/train/images',   # Source folder for training images
    'train_labels': 'source_datasets/datasets/train/labels',   # Source folder for training labels
    'valid_images': 'source_datasets/datasets/valid/images',   # Source folder for validation images
    'valid_labels': 'source_datasets/datasets/valid/labels'    # Source folder for validation labels
}

# Define the destination folder structure as per the desired format
destination_folder = {
    'train_images': 'datasets/train/images',    # Destination folder for training images
    'val_images': 'datasets/train/val',         # Destination folder for validation images
    'train_labels': 'datasets/labels/train',    # Destination folder for training labels
    'val_labels': 'datasets/labels/val'         # Destination folder for validation labels
}

# Function to move files from the source folder to the destination folder
def move_files(source, destination):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # Get a list of all files in the source folder
    files = os.listdir(source)
    
    # Use tqdm to show a progress bar while moving files
    for filename in tqdm(files, desc=f"Moving files from {source} to {destination}"):
        file_path = os.path.join(source, filename)      # Define the full path of the source file
        dest_path = os.path.join(destination, filename) # Define the full path of the destination file
        
        try:
            # If the file exists and is valid, move it to the destination folder
            if os.path.isfile(file_path):
                shutil.move(file_path, dest_path)
                # Show a success warning if the file is moved successfully
                warnings.warn(f"Successfully moved {filename} to {destination}", UserWarning)
            else:
                # Show a warning if the file is not valid
                warnings.warn(f"{filename} is not a valid file", UserWarning)
        except Exception as e:
            # Show a warning if there was an error moving the file
            warnings.warn(f"Failed to move {filename} due to: {str(e)}", UserWarning)

# Move images and labels from train and valid sets with progress bar
# Moving image and label files from Roboflow folder structure to the new desired format.
move_files(source_folder['train_images'], destination_folder['train_images'])
move_files(source_folder['train_labels'], destination_folder['train_labels'])
move_files(source_folder['valid_images'], destination_folder['val_images'])
move_files(source_folder['valid_labels'], destination_folder['val_labels'])
