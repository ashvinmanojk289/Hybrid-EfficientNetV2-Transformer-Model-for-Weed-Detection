import os
import json
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(base_dir, output_dir):
    """
    Combines images from train, valid, and test folders, reads annotations,
    and splits them into new train, validation, and test sets.

    Args:
        base_dir (str): Path to the directory containing the original dataset.
        output_dir (str): Path to the directory where the prepared dataset will be saved.
    """
    print("Starting dataset preparation...")
    # Create output directories
    for split in ['train', 'valid', 'test']:
        for class_name in ['weed', 'crop']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    all_images = []
    all_labels = []

    # Process each split (train, valid, test) from the original dataset
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, split)
        annotations_path = os.path.join(split_dir, '_annotations.coco.json')

        # Check if annotation file exists
        if not os.path.exists(annotations_path):
            print(f"Warning: Annotation file not found at {annotations_path}. Skipping this split.")
            continue

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Create a mapping from image_id to file_name
        image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

        # Process annotations to get image paths and labels
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id'] # 0 for crop, 1 for weed
            class_name = 'weed' if category_id == 1 else 'crop'

            if image_id in image_id_to_filename:
                filename = image_id_to_filename[image_id]
                src_path = os.path.join(split_dir, filename)
                if os.path.exists(src_path):
                    all_images.append(src_path)
                    all_labels.append(class_name)
                else:
                    print(f"Warning: Image file not found at {src_path}")

    print(f"Total images found: {len(all_images)}")
    if not all_images:
        print("Error: No images were found. Please check the dataset structure and paths.")
        return

    # Split the data into training (70%) and temp (30%)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels)

    # Split the temp data into validation (50% of temp -> 15% of total) and test (50% of temp -> 15% of total)
    valid_images, test_images, valid_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(valid_images)}")
    print(f"Test set size: {len(test_images)}")

    # Function to copy files to the new structure
    def copy_files(images, labels, split_name):
        print(f"Copying files for {split_name} set...")
        for img_path, label in zip(images, labels):
            dest_dir = os.path.join(output_dir, split_name, label)
            shutil.copy(img_path, dest_dir)

    # Copy files to their respective directories
    copy_files(train_images, train_labels, 'train')
    copy_files(valid_images, valid_labels, 'valid')
    copy_files(test_images, test_labels, 'test')

    print("Dataset preparation complete.")
    print(f"Prepared dataset saved at: {output_dir}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Update this path to the root directory of your downloaded dataset
    # e.g., 'C:/Users/YourUser/Downloads/Weed-Detection'
    ORIGINAL_DATASET_DIR = 'path/to/your/Weed-Detection'

    # Directory where the newly structured dataset will be stored
    PREPARED_DATASET_DIR = 'prepared_dataset'

    if not os.path.exists(ORIGINAL_DATASET_DIR) or ORIGINAL_DATASET_DIR == 'path/to/your/Weed-Detection':
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! ERROR: Please update ORIGINAL_DATASET_DIR in       !!!")
         print("!!! prepare_data.py to point to your dataset location. !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        prepare_dataset(ORIGINAL_DATASET_DIR, PREPARED_DATASET_DIR)
