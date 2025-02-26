import os
import random
import shutil
from collections import defaultdict

def count_instances(source_folder, classes):
    """
    Counts the number of instances for each class across all images.
    """
    instance_counts = {cls: 0 for cls in classes}
    image_annotations = {}

    # List all annotation files
    files = os.listdir(source_folder)
    annotation_files = [f for f in files if f.endswith('.txt')]

    for annotation_file in annotation_files:
        base_name = os.path.splitext(annotation_file)[0]
        txt_path = os.path.join(source_folder, annotation_file)
        img_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            if os.path.exists(os.path.join(source_folder, base_name + ext)):
                img_found = True
                break
        if not img_found:
            continue  # Skip if no corresponding image file is found

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            annotations = []
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    cls_id = int(parts[0])
                    if cls_id in classes:
                        instance_counts[cls_id] += 1
                        annotations.append((cls_id, parts[1:]))
            if annotations:
                image_annotations[base_name] = annotations

    return instance_counts, image_annotations

def select_images_per_class(instance_counts, image_annotations, ipc, classes):
    """
    Selects images per class to accumulate at least ipc instances for each class.
    """
    selected_images = set()
    class_instance_accum = {cls: 0 for cls in classes}
    class_image_map = defaultdict(list)

    # Build a map of class to images that contain that class
    for image_name, annotations in image_annotations.items():
        classes_in_image = set(cls_id for cls_id, _ in annotations)
        for cls_id in classes_in_image:
            class_image_map[cls_id].append(image_name)

    # For each class, shuffle the list of images that contain the class
    for cls_id in classes:
        image_list = list(set(class_image_map[cls_id]))
        random.shuffle(image_list)
        accum_instances = 0
        image_idx = 0
        while accum_instances < ipc and image_idx < len(image_list):
            image_name = image_list[image_idx]
            if image_name not in selected_images:
                selected_images.add(image_name)
                # Count instances of this class in the image
                instances_in_image = sum(1 for anno_cls_id, _ in image_annotations[image_name] if anno_cls_id == cls_id)
                accum_instances += instances_in_image
            else:
                # If the image is already selected, we may still get more instances of the class
                instances_in_image = sum(1 for anno_cls_id, _ in image_annotations[image_name] if anno_cls_id == cls_id)
                accum_instances += instances_in_image
            image_idx += 1
        class_instance_accum[cls_id] = accum_instances
        if accum_instances < ipc:
            print(f"Warning: Could not accumulate {ipc} instances for class {cls_id}. Accumulated {accum_instances} instances.")

    return selected_images

def copy_selected_images(selected_images, source_folder, destination_folder):
    """
    Copies the selected images and their annotation files to the destination folder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for base_name in selected_images:
        # Copy image file
        src_image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            potential_path = os.path.join(source_folder, base_name + ext)
            if os.path.exists(potential_path):
                src_image_path = potential_path
                break  # Found the image file
        if src_image_path:
            dst_image_path = os.path.join(destination_folder, os.path.basename(src_image_path))
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"Image file for {base_name} not found. Skipping.")

        # Copy annotation file
        src_txt_path = os.path.join(source_folder, base_name + '.txt')
        if os.path.exists(src_txt_path):
            dst_txt_path = os.path.join(destination_folder, base_name + '.txt')
            shutil.copy2(src_txt_path, dst_txt_path)
        else:
            print(f"Annotation file for {base_name} not found. Skipping.")

def main():
    # Set the source and destination folders
    source_folder = './train'  # Replace with your source folder path
    destination_folder = './quanti_charSeg'  # Replace with your destination folder path

    # Define the classes (assuming classes are labeled from 0 to 25 for letters A-Z)
    classes = list(range(1))  # [0, 1, 2, ..., 25]

    # Count instances per class and collect image annotations
    instance_counts, image_annotations = count_instances(source_folder, classes)

    # Find the smallest number of instances per class (ipc)
    ipc = min(instance_counts.values())
    ipc = 1500
    print(f"The smallest number of instances per class is {ipc}.")

    # Select images to accumulate at least ipc instances per class
    selected_images = select_images_per_class(instance_counts, image_annotations, ipc, classes)
    print(f"Total selected images: {len(selected_images)}.")

    # Copy the selected images and their annotations
    copy_selected_images(selected_images, source_folder, destination_folder)

    print("Images have been successfully copied.")

if __name__ == '__main__':
    main()
