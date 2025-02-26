import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w, num_anchors, _ = input.shape

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(input[..., 5:])
    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])

    anchors = np.array(anchors)
    box_wh *= anchors.reshape(1, 1, len(mask), 2)

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_h, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(grid_h, 1), grid_w)
    col = col.reshape(grid_h, grid_w, 1, 1)
    row = row.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    input_shape_w, input_shape_h = 416, 448
    box_wh /= (input_shape_w, input_shape_h)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= obj_thresh)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, nms_thresh):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1)
        h1 = np.maximum(0.0, yy2 - yy1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

def yolov3_post_process(input_data, anchors, masks, obj_thresh, nms_thresh):
    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s, obj_thresh)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    if not boxes:
        return None, None, None

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thresh)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def process_char_seg(input, mask, anchors):
    """Specific process function for character segmentation"""
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w, num_anchors, _ = input.shape

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(input[..., 5:])
    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])

    anchors = np.array(anchors)
    box_wh *= anchors.reshape(1, 1, len(mask), 2)

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_h, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(grid_h, 1), grid_w)
    col = col.reshape(grid_h, grid_w, 1, 1)
    row = row.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    # Use the correct input dimensions for char segmentation
    input_shape_w, input_shape_h = 256, 96  # Changed from 416, 448
    box_wh /= (input_shape_w, input_shape_h)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def yolov3_post_process_char_seg(input_data, anchors, masks, obj_thresh, nms_thresh):
    """Specific post-process function for character segmentation"""
    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process_char_seg(input, mask, anchors)  # Use the char-specific process function
        b, c, s = filter_boxes(b, c, s, obj_thresh)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    if not boxes:
        return None, None, None

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thresh)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # Sort boxes from left to right
    indices = np.argsort(boxes[:, 0])
    boxes = boxes[indices]
    classes = classes[indices]
    scores = scores[indices]

    return boxes, classes, scores