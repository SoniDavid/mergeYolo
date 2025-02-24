import os
import shutil
import yaml
from pathlib import Path
import sys

def merge_yolo_datasets(original_dataset_dir, new_dataset_dir, output_dir, class_offset=None):
    """
    Merge two YOLOv8 datasets, adjusting class indices for the new dataset.
    
    Args:
        original_dataset_dir: Path to the original (larger) dataset
        new_dataset_dir: Path to the new dataset with classes to be added
        output_dir: Path where the merged dataset will be stored
        class_offset: Number of classes in the original dataset (will be auto-detected if None)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directories for train, valid, test splits
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Load class information from data.yaml files
    original_yaml_path = os.path.join(original_dataset_dir, 'data.yaml')
    new_yaml_path = os.path.join(new_dataset_dir, 'data.yaml')
    
    # Verify YAML files exist
    if not os.path.exists(original_yaml_path):
        print(f"ERROR: YAML file not found at {original_yaml_path}")
        print(f"Directory contents: {os.listdir(original_dataset_dir)}")
        return False
    
    if not os.path.exists(new_yaml_path):
        print(f"ERROR: YAML file not found at {new_yaml_path}")
        print(f"Directory contents: {os.listdir(new_dataset_dir)}")
        return False
    
    with open(original_yaml_path, 'r') as f:
        original_data = yaml.safe_load(f)
    
    with open(new_yaml_path, 'r') as f:
        new_data = yaml.safe_load(f)
    
    # Get class information
    original_classes = original_data.get('names', {})
    new_classes = new_data.get('names', {})
    
    # If class_offset wasn't provided, calculate it
    if class_offset is None:
        if isinstance(original_classes, list):
            class_offset = len(original_classes)
        else:  # If it's a dict
            class_offset = max(map(int, original_classes.keys())) + 1
    
    print(f"Original dataset has {class_offset} classes")
    print(f"Adding {len(new_classes)} new classes")
    
    # Create merged class list
    merged_classes = original_classes.copy() if isinstance(original_classes, dict) else {i: name for i, name in enumerate(original_classes)}
    
    if isinstance(new_classes, list):
        for i, class_name in enumerate(new_classes):
            merged_classes[class_offset + i] = class_name
    else:  # If it's a dict
        for class_idx, class_name in new_classes.items():
            merged_classes[class_offset + int(class_idx)] = class_name
    
    # Process original dataset
    process_dataset(original_dataset_dir, output_dir, modify_classes=False)
    
    # Process new dataset with class adjustments
    process_dataset(new_dataset_dir, output_dir, modify_classes=True, 
                  class_offset=class_offset, prefix="new_")
    
    # Create merged data.yaml
    merged_data = original_data.copy()
    merged_data['names'] = merged_classes
    
    # Update paths in the merged data.yaml
    merged_data['train'] = './train/images'
    merged_data['val'] = './valid/images'
    merged_data['test'] = './test/images'
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(merged_data, f, sort_keys=False)
    
    print(f"Dataset successfully merged into {output_dir}")
    print(f"Final class count: {len(merged_classes)}")
    return merged_classes

def process_dataset(dataset_dir, output_dir, modify_classes=False, class_offset=0, prefix=""):
    """
    Process dataset files, optionally modifying class indices.
    Maintains the train/valid/test split structure.
    """
    # Handle train, valid, and test splits
    splits = ['train', 'valid', 'test']
    
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
                
            # Generate new filename to avoid conflicts
            new_img_filename = f"{prefix}{img_file}"
            
            # Copy the image file
            shutil.copy2(
                os.path.join(images_dir, img_file),
                os.path.join(output_images, new_img_filename)
            )
            
            # Get the corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                new_label_filename = f"{prefix}{label_file}"
                new_label_path = os.path.join(output_labels, new_label_filename)
                
                if modify_classes:
                    # Modify class indices
                    with open(label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # Valid YOLO format line
                                class_idx = int(parts[0])
                                new_class_idx = class_idx + class_offset
                                parts[0] = str(new_class_idx)
                                f_out.write(' '.join(parts) + '\n')
                else:
                    # Just copy the label file without modifications
                    shutil.copy2(label_path, new_label_path)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge two YOLOv8 datasets')
    parser.add_argument('--original', required=True, help='Path to original (larger) dataset')
    parser.add_argument('--new', required=True, help='Path to new dataset with classes to add')
    parser.add_argument('--output', required=True, help='Path for merged dataset')
    parser.add_argument('--class-offset', type=int, help='Number of classes in original dataset (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Ensure absolute paths
    original_path = os.path.abspath(args.original)
    new_path = os.path.abspath(args.new)
    output_path = os.path.abspath(args.output)
    
    print(f"Original dataset: {original_path}")
    print(f"New dataset: {new_path}")
    print(f"Output path: {output_path}")
    
    merged_classes = merge_yolo_datasets(
        original_path,
        new_path,
        output_path,
        args.class_offset
    )
    
    if merged_classes:
        print("\nMerged class mapping:")
        for idx, name in sorted(merged_classes.items()):
            print(f"{idx}: {name}")
    else:
        print("\nDataset merging failed. Please check the error messages above.")
        sys.exit(1)