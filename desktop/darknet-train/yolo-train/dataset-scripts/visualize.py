import cv2
import os

def draw_boxes(image_path, label_path, class_names):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        class_id, x, y, bw, bh = map(float, parts)

        x, y, bw, bh = int(x * w), int(y * h), int(bw * w), int(bh * h)
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_names[int(class_id)], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Bounding Boxes", img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

# Example usage
class_names = ["A", "B", "C", ..., "Z"]  # Define all 26 classes
draw_boxes("path/to/image.png", "path/to/image.txt", class_names)
