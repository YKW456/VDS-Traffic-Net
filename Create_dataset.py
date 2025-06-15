import os
import cv2
import numpy as np
import warnings
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Fix np.bool issue
if not hasattr(np, 'bool'):
    np.bool = bool
    warnings.warn("Patched np.bool to bool for imgaug compatibility", RuntimeWarning)
from imgaug import augmenters as iaa
from generate_jiguang_model import *


def resize_and_grayscale_images(target_size=(224, 224)):
    """
    Resize all images in input directory to target size and convert to grayscale, then save to output directory

    Parameters:
        input_dir: Input image directory
        output_dir: Output image directory
        target_size: Target size (width, height)
    """
    input_dir = "Origin_photo"
    output_dir = "Resized_gray_dataset"  # Modified output directory name to reflect processing

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print(f"Starting to process {len(image_files)} images...")

    # Process each image
    for filename in tqdm(image_files, desc="Resizing and converting to grayscale"):
        try:
            # Read image (compatible with Chinese paths)
            img_path = os.path.join(input_dir, filename)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                print(f"Cannot read: {img_path}")
                continue

            # Resize image
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            # Save image (keep original filename and format)
            output_path = os.path.join(output_dir, filename)
            ext = os.path.splitext(filename)[1].lower()

            # Choose save format based on original file extension
            if ext in ['.jpg', '.jpeg']:
                cv2.imencode(ext, gray_img)[1].tofile(output_path)
            elif ext == '.png':
                cv2.imencode(ext, gray_img)[1].tofile(output_path)
            else:
                # Default to JPEG
                cv2.imencode('.jpg', gray_img)[1].tofile(output_path)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print(f"All images processed! Results saved in {output_dir}")


def get_augmenter():
    return iaa.Sometimes(
        0.8,  # 80% probability to apply augmentation
        iaa.OneOf([
            iaa.Crop(percent=(0, 0.2)),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-5, 5)
            ),
            iaa.GaussianBlur(sigma=(0, 1)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.08 * 255)),
            iaa.Multiply((0.7, 1.3)),
            iaa.LinearContrast((0.7, 1.3))
        ])
    )


def process_single_file(file_info, temp_dir, target_dir, total_len):
    file, root = file_info
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        return

    img_path = os.path.join(root, file)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot read: {img_path}")
        return

    # Get label (first three characters)
    label = file[:-4] if len(file) >= 3 else "000"

    # Create target subdirectory
    label_dir = os.path.join(target_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Generate 50 augmented versions
    for i in range(total_len):
        try:
            # Create new augmenter instance each time to avoid random state issues
            augmented_img = get_augmenter().augment_image(img)

            # Generate English-only filename
            new_filename = f"{label}_{i:02d}.png"
            temp_path = os.path.join(temp_dir, new_filename)
            output_path = os.path.join(label_dir, new_filename)

            # Save image (compatible with all paths)
            cv2.imencode('.png', augmented_img)[1].tofile(temp_path)

            # Copy directly to target directory
            shutil.copy2(temp_path, output_path)

        except Exception as e:
            print(f"Error processing {file} during {i}th augmentation: {str(e)}")


def process_images(dataset_size):
    # Create output directory
    target_dir = "dataset_train"
    # Save to the above path
    temp_dir = "dataset_temp"

    # Ensure directories exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    # Collect all files to process
    file_list = []
    for root, dirs, files in os.walk("Resized_gray_dataset"):
        for file in files:
            file_list.append((file, root))

    # Use multiprocessing
    num_processes = max(1, cpu_count() - 1)  # Leave one core for system
    print(f"Using {num_processes} processes for parallel processing...")

    with Pool(num_processes) as pool:
        # Use partial to fix some parameters
        worker_func = partial(process_single_file, temp_dir=temp_dir, target_dir=target_dir, total_len=dataset_size)
        # Use tqdm to show progress
        list(tqdm(pool.imap(worker_func, file_list), total=len(file_list), desc="Processing images"))

    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning temp directory: {str(e)}")

    print(f"All images processed! Results saved in {target_dir}")


def convert_to_grayscale_and_save():
    path1 = 'classification-jiguang'
    path2 = 'classification-gray'

    # Ensure path2 exists
    if not os.path.exists(path2):
        os.makedirs(path2)
    # Walk through all files and subdirectories in path1
    for root, dirs, files in os.walk(path1):
        # Calculate corresponding path in path2
        relative_path = os.path.relpath(root, path1)
        dest_dir = os.path.join(path2, relative_path)

        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Process all files in current directory
        for file in tqdm(files, desc=f'Processing {relative_path}'):
            file_path = os.path.join(root, file)

            # Try to read image file
            try:
                img = cv2.imread(file_path)
                if img is not None:  # If valid image file
                    # Convert to grayscale
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Save to corresponding location in path2
                    dest_path = os.path.join(dest_dir, file)
                    cv2.imwrite(dest_path, gray_img)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    glow_size_range = (1335, 2225)
    center_brightness = 255
    input_dir = "Origin_photo"
    dataset_size = 10
    for i, filename in enumerate(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, filename)
        print(input_path)
        generate_0(input_path, dataset_size, glow_size_range, center_brightness)
    # Generate 14*dataset_size*8
    convert()
    # Generate 8*dataset_size*14
    convert_to_grayscale_and_save()
    # Generate 14*dataset_size*8 gray test set
    resize_and_grayscale_images(target_size=(224, 224))
    process_images(dataset_size)
    # Generate 14*dataset_size*8 train set