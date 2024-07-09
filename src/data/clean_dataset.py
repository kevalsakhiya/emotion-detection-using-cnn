import os
import imghdr
import cv2
from typing import NoReturn

# Define the list of acceptable image extensions
IMAGE_EXTS = ['jpeg', 'jpg', 'png']

def remove_invalid_images(data_dir: str) -> NoReturn:
    """
    Walk through the directory and remove files that are not valid images.
    
    Parameters:
    data_dir (str): Path to the directory containing image classes and possibly other nested subdirectories
    """
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            process_image(file_path)

def process_image(file_path: str) -> NoReturn:
    """
    Process and validate an image file.
    
    Parameters:
    file_path (str): Path to the image file
    
    """
    try:
        file_type = imghdr.what(file_path)
        
        if file_type not in IMAGE_EXTS:
            print(f'Image not in ext list: {file_path}')
            os.remove(file_path)
        else:
            img = cv2.imread(file_path)
            
    except Exception as e:
        handle_error(file_path, e)

def handle_error(file_path: str, error: Exception) -> NoReturn:
    """
    Handle errors encountered while processing files.
    
    Parameters:
    file_path (str): Path to the problematic file
    error (Exception): The exception that was raised
    """
    print(f'Issue with file {file_path}. Error: {error}')
    os.remove(file_path)

if __name__ == '__main__':
    DATA_DIR = '../../data/raw'
    remove_invalid_images(DATA_DIR)
