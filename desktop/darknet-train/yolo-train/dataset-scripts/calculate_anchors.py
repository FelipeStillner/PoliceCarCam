from sklearn.cluster import KMeans

def iou(box, clusters):
    """
    Calculate Intersection over Union (IoU) between a box and k clusters.
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def avg_iou(boxes, clusters):
    """
    Calculate the average IoU between all boxes and k clusters.
    """
    return np.mean([np.max(iou(box, clusters)) for box in boxes])

def kmeans(boxes, k, max_iter=100):
    """
    Run k-means clustering to find k anchors.
    """
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iter)
    kmeans.fit(boxes)
    return kmeans.cluster_centers_

import os
import numpy as np

def parse_annotations(labels_path,width,heigh):
    bboxes = []
    for label_file in os.listdir(labels_path):
        if label_file.endswith(".txt"):
            with open(os.path.join(labels_path, label_file), "r") as f:
                for line in f:
                    # _, _, _, w, h = map(float, line.strip().split())
                    parts = line.strip().split()
                    _, _, w, h = map(float, parts[1:])  # Skip the first value
                    bboxes.append((w * width, h * heigh))  # Scale to input size
    return np.array(bboxes)


# labels_path='../characterSeg/dataset/train/'

# labels_path='../charRec/crop/train/' #16, 58, 19, 67, 22, 75, 26, 84, 32, 96

# labels_path='../digitRec/crop/train/' #15, 58, 18, 67, 22, 75, 26, 84, 32, 96
labels_path='../characterSeg/dataset/train/' # 3, 43, 5, 31, 3, 49, 4, 54, 4, 61
#1, 2, 1, 3, 1, 5, 2, 3, 3, 5 -lp rec

bboxes = parse_annotations(labels_path,32,96)
print(f"Extracted {len(bboxes)} bounding boxes.")

# Number of anchors
num_anchors = 5

# Run k-means clustering
anchors = kmeans(bboxes, k=num_anchors)
anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # Sort by area
print(f"Calculated Anchors:\n{anchors}")


formatted_anchors = ", ".join([f"{round(w, 2)},{round(h, 2)}" for w, h in anchors])
print(f"Anchors for YOLO:\n{formatted_anchors}")

# Format anchors for Darknet
anchor_str = ", ".join([f"{w},{h}" for w, h in anchors.flatten().reshape(-1, 2)])
print(f"New anchor string for cfg file: {anchor_str}")


import numpy as np

# Example float anchors from k-means
float_anchors = [3.33,43.04, 5.29,31.26, 3.41,48.55, 3.51,53.7, 3.65,61.24]

# Convert to integers
int_anchors = [round(a) for a in float_anchors]

# Format for YOLO
anchor_str = ", ".join(map(str, int_anchors))
print("Updated anchors:", anchor_str)

# 0.62,1.38, 1.05,1.03, 1.52,1.48, 2.07,2.01, 2.75,2.67
