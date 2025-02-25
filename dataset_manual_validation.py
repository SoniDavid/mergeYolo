import os
import glob

# Paths to your images and labels
img_dir = '/home/soni/yoloLocalRun/runYolo/train/images'
label_dir = '/home/soni/yoloLocalRun/runYolo/train/labels'

# Get all image files
img_files = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

# Count variables
total_images = len(img_files)
missing_labels = 0

print(f"Checking {total_images} images for corresponding labels...")

# Check all images
for img_path in img_files:
    # Get corresponding label file
    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
    label_path = os.path.join(label_dir, base_name + '.txt')
    
    # Check if label exists
    if not os.path.exists(label_path):
        print(f"Missing label for: {base_name}")
        missing_labels += 1

# Summary
print(f"\nSummary:")
print(f"Total images: {total_images}")
print(f"Images missing labels: {missing_labels}")
print(f"Percentage complete: {((total_images - missing_labels) / total_images) * 100:.2f}%")

# Also check for orphaned label files (labels without images)
label_files = glob.glob(os.path.join(label_dir, '*.txt'))
orphaned_labels = 0

for label_path in label_files:
    base_name = os.path.basename(label_path).rsplit('.', 1)[0]
    
    # Check for corresponding image (both jpg and png)
    jpg_path = os.path.join(img_dir, base_name + '.jpg')
    png_path = os.path.join(img_dir, base_name + '.png')
    
    if not (os.path.exists(jpg_path) or os.path.exists(png_path)):
        print(f"Orphaned label without image: {base_name}")
        orphaned_labels += 1

print(f"Orphaned labels (without images): {orphaned_labels}")