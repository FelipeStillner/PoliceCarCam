import os
import shutil
import random

def collect_car_images(source_folder):
    """
    Collects a mapping from each car to the list of its image files.
    """
    car_images = {}
    files = os.listdir(source_folder)
    
    for file_name in files:
        if os.path.isfile(os.path.join(source_folder, file_name)):
            # Extract car ID from the file name
            if file_name.startswith('track') and '[' in file_name and ']' in file_name:
                base_name = file_name.split('[')[0]  # 'trackXXXX'
                car_id = base_name[len('track'):]
                if car_id.isdigit():
                    if car_id not in car_images:
                        car_images[car_id] = []
                    car_images[car_id].append(file_name)
                else:
                    print(f"Warning: Unable to extract car ID from file name {file_name}")
            else:
                print(f"Warning: File name {file_name} does not match expected format. Skipping.")
    return car_images

def main():
    # Set the source and destination folders
    source_folder = './train'  # Replace with your source folder path
    destination_folder = './quanti_lpRec'  # Replace with your destination folder path

    # Collect car images
    car_images = collect_car_images(source_folder)

    # Find the minimum number of images any car has
    min_num_images = min(len(images) for images in car_images.values())
    min_num_images = 3
    print(f"The smallest number of images per car is {min_num_images}.")

    # Copy min_num_images images for each car to the destination folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for car_id, images in car_images.items():
        # Sort images or shuffle them if you prefer random selection
        images.sort()  # Or use random.shuffle(images) for random selection
        selected_images = images[:min_num_images]

        # Copy selected images to the destination folder
        for image_file in selected_images:
            src_image_path = os.path.join(source_folder, image_file)
            dst_image_path = os.path.join(destination_folder, image_file)
            shutil.copy2(src_image_path, dst_image_path)

    print("Images have been successfully copied.")

if __name__ == '__main__':
    main()
