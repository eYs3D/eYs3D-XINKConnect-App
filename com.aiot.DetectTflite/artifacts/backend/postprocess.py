"""
YOLO11 detection post-processing (pure numpy).

Targets our per-tensor INT8 TFLite export (Ultralytics SavedModel export +
strict per-tensor requant — see tools/convert_to_tflite.py): a single output
tensor of shape (1, 4+nc, N) or (1, N, 4+nc) where box coords are xywh
**normalized to [0,1]** relative to the model input size and class scores are
already activated. Box and class scores share one per-tensor output scale, so
box MUST stay normalized (pixel-space box would collapse the class scores onto
the quantization zero-point); the decoder multiplies box by input_size here.

Kept dependency-free (numpy only) so it is unit-testable without cv2 / tflite.
The interpreter call and letterbox image op live in main.py / tflite_backend.py.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


def xywh2xyxy(b):
    """(N,4) center xywh → corner xyxy."""
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2.0
    out[:, 1] = b[:, 1] - b[:, 3] / 2.0
    out[:, 2] = b[:, 0] + b[:, 2] / 2.0
    out[:, 3] = b[:, 1] + b[:, 3] / 2.0
    return out


def nms(boxes, scores, iou_thr):
    """Greedy NMS, pure numpy. boxes are xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def nms_classwise(boxes, scores, classes, iou_thr):
    """Run NMS independently per class. Returns kept indices (into input)."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        k = nms(boxes[idx], scores[idx], iou_thr)
        keep.extend(idx[k].tolist())
    return keep


def scale_boxes_back(boxes, lb, src_w, src_h):
    """Map xyxy boxes from letterboxed pixel space back to original image."""
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - lb["pad_x"]) / lb["scale"]
    out[:, [1, 3]] = (out[:, [1, 3]] - lb["pad_y"]) / lb["scale"]
    np.clip(out[:, [0, 2]], 0, src_w, out=out[:, [0, 2]])
    np.clip(out[:, [1, 3]], 0, src_h, out=out[:, [1, 3]])
    return out


def _to_anchors_channels(pred):
    """Squeeze batch and orient to (N_anchors, C). Ultralytics emits (C, N) with
    C = 4+nc << N; transpose when the first axis is the smaller (channels) one."""
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    return pred


def decode_detect(pred, conf_thr, iou_thr, input_size, lb, src_w, src_h,
                  classes_filter=None):
    """Decode a YOLO11 detect head into (boxes_xyxy, scores, classes).

    boxes_xyxy are in ORIGINAL image coordinates. Returns empty arrays when no
    detection passes the threshold.
    """
    pred = _to_anchors_channels(pred)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    classes = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    mask = confs >= conf_thr
    if classes_filter is not None:
        mask &= np.isin(classes, list(classes_filter))
    if not mask.any():
        return (np.empty((0, 4), np.float32),
                np.empty(0, np.float32),
                np.empty(0, np.int32))

    boxes_xywh = boxes_xywh[mask] * float(input_size)  # normalized → input px
    confs = confs[mask]
    classes = classes[mask].astype(np.int32)
    boxes = xywh2xyxy(boxes_xywh)

    keep = nms_classwise(boxes, confs, classes, iou_thr)
    boxes = scale_boxes_back(boxes[keep], lb, src_w, src_h)
    return boxes, confs[keep], classes[keep]
