import os
import shutil
import yaml
from pathlib import Path
import sys
import argparse

def remove_yolo_classes(dataset_dir, output_dir, classes_to_remove):
    """
    Remove specified classes from a YOLOv8 dataset.
    
    Args:
        dataset_dir: Path to the original dataset
        output_dir: Path where the modified dataset will be stored
        classes_to_remove: List of class indices or names to remove
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directories for train, valid, test splits
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Load class information from data.yaml file
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    # Verify YAML file exists
    if not os.path.exists(yaml_path):
        print(f"ERROR: YAML file not found at {yaml_path}")
        print(f"Directory contents: {os.listdir(dataset_dir)}")
        return False
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get class information
    classes = data.get('names', {})
    
    # Convert class names to indices if needed
    class_indices_to_remove = []
    
    for class_item in classes_to_remove:
        if isinstance(class_item, int) or class_item.isdigit():
            class_indices_to_remove.append(int(class_item))
        else:
            # Find the index of the class name
            if isinstance(classes, list):
                if class_item in classes:
                    class_indices_to_remove.append(classes.index(class_item))
            else:  # If it's a dict
                for idx, name in classes.items():
                    if name == class_item:
                        class_indices_to_remove.append(int(idx))
    
    print(f"Removing classes with indices: {class_indices_to_remove}")
    
    # Create updated class mapping
    updated_classes = {}
    class_mapping = {}  # Maps old indices to new indices
    
    if isinstance(classes, list):
        new_idx = 0
        for old_idx, class_name in enumerate(classes):
            if old_idx not in class_indices_to_remove:
                updated_classes[new_idx] = class_name
                class_mapping[old_idx] = new_idx
                new_idx += 1
    else:  # If it's a dict
        new_idx = 0
        for old_idx, class_name in sorted(classes.items(), key=lambda x: int(x[0])):
            old_idx = int(old_idx)
            if old_idx not in class_indices_to_remove:
                updated_classes[new_idx] = class_name
                class_mapping[old_idx] = new_idx
                new_idx += 1
    
    # Process the dataset
    process_dataset(dataset_dir, output_dir, class_indices_to_remove, class_mapping)
    
    # Create updated data.yaml
    updated_data = data.copy()
    updated_data['names'] = updated_classes
    
    # Update paths in the updated data.yaml
    updated_data['train'] = './train/images'
    updated_data['val'] = './valid/images'
    updated_data['test'] = './test/images'
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(updated_data, f, sort_keys=False)
    
    print(f"Dataset successfully modified and saved to {output_dir}")
    print(f"Original class count: {len(classes)}")
    print(f"Updated class count: {len(updated_classes)}")
    
    return updated_classes

def process_dataset(dataset_dir, output_dir, class_indices_to_remove, class_mapping):
    """
    Process dataset files, removing specified classes and remapping remaining ones.
    Maintains the train/valid/test split structure.
    """
    # Handle train, valid, and test splits
    splits = ['train', 'valid', 'test']
    
    processed_counts = {
        'total_images': 0,
        'removed_images': 0,
        'modified_labels': 0
    }
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            continue
        
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            continue
            
        if not os.path.exists(labels_dir):
            print(f"Labels directory not found: {labels_dir}")
            continue
        
        print(f"Processing {split} split...")
        
        # Output directories for this split
        output_images = os.path.join(output_dir, split, 'images')
        output_labels = os.path.join(output_dir, split, 'labels')
        
        # Process all image files
        for img_file in os.listdir(images_dir):
            if not (img_file.endswith('.jpg') or img_file.endswith('.jpeg') or 
                    img_file.endswith('.png') or img_file.endswith('.bmp')):
                continue
            
            processed_counts['total_images'] += 1
            
            # Get the corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            keep_image = True
            has_valid_annotations = False
            
            if os.path.exists(label_path):
                new_label_path = os.path.join(output_labels, label_file)
                
                # Process label file
                with open(label_path, 'r') as f_in:
                    lines = f_in.readlines()
                
                # Check if all annotations belong to classes we want to remove
                if lines:
                    has_annotations_for_other_classes = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Valid YOLO format line
                            class_idx = int(parts[0])
                            if class_idx not in class_indices_to_remove:
                                has_annotations_for_other_classes = True
                                break
                    
                    if not has_annotations_for_other_classes:
                        keep_image = False
                        processed_counts['removed_images'] += 1
                
                if keep_image:
                    # Write new label file with filtered and remapped classes
                    with open(new_label_path, 'w') as f_out:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # Valid YOLO format line
                                class_idx = int(parts[0])
                                if class_idx not in class_indices_to_remove:
                                    # Remap class index
                                    parts[0] = str(class_mapping[class_idx])
                                    f_out.write(' '.join(parts) + '\n')
                                    has_valid_annotations = True
                    
                    if has_valid_annotations:
                        processed_counts['modified_labels'] += 1
                    else:
                        # If after filtering there are no annotations left, don't keep the image
                        keep_image = False
                        processed_counts['removed_images'] += 1
                        if os.path.exists(new_label_path):
                            os.remove(new_label_path)
            
            # Copy the image file if we're keeping it
            if keep_image:
                shutil.copy2(
                    os.path.join(images_dir, img_file),
                    os.path.join(output_images, img_file)
                )
    
    print(f"Total images processed: {processed_counts['total_images']}")
    print(f"Images removed (containing only specified classes): {processed_counts['removed_images']}")
    print(f"Images with modified labels: {processed_counts['modified_labels']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove specified classes from a YOLOv8 dataset')
    parser.add_argument('--dataset', required=True, help='Path to the original dataset')
    parser.add_argument('--output', required=True, help='Path for the modified dataset')
    parser.add_argument('--remove-classes', required=True, nargs='+', 
                        help='List of class indices or names to remove, space-separated')
    
    args = parser.parse_args()
    
    # Ensure absolute paths
    dataset_path = os.path.abspath(args.dataset)
    output_path = os.path.abspath(args.output)
    
    print(f"Original dataset: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Classes to remove: {args.remove_classes}")
    
    updated_classes = remove_yolo_classes(
        dataset_path,
        output_path,
        args.remove_classes
    )
    
    if updated_classes:
        print("\nUpdated class mapping:")
        for idx, name in sorted(updated_classes.items()):
            print(f"{idx}: {name}")
    else:
        print("\nClass removal failed. Please check the error messages above.")
        sys.exit(1)
