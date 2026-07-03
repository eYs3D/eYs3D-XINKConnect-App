"""
YOLO11 detect + classify post-processing for CarClassifyTflite (pure numpy).

Two models:
  detect   → per-tensor INT8 detect head (1, 4+nc, N), xywh NORMALIZED [0,1]
             (box and class scores share one per-tensor output scale; the
             decoder multiplies box by input_size — see DetectTflite postprocess)
  classify → (1, ncls) probabilities (already softmaxed by Ultralytics export)

Numpy-only so it is unit-testable without cv2 / tflite.
"""

import numpy as np


def xywh2xyxy(b):
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2.0
    out[:, 1] = b[:, 1] - b[:, 3] / 2.0
    out[:, 2] = b[:, 0] + b[:, 2] / 2.0
    out[:, 3] = b[:, 1] + b[:, 3] / 2.0
    return out


def nms(boxes, scores, iou_thr):
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
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        k = nms(boxes[idx], scores[idx], iou_thr)
        keep.extend(idx[k].tolist())
    return keep


def scale_boxes_back(boxes, lb, src_w, src_h):
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - lb["pad_x"]) / lb["scale"]
    out[:, [1, 3]] = (out[:, [1, 3]] - lb["pad_y"]) / lb["scale"]
    np.clip(out[:, [0, 2]], 0, src_w, out=out[:, [0, 2]])
    np.clip(out[:, [1, 3]], 0, src_h, out=out[:, [1, 3]])
    return out


def _to_anchors_channels(pred):
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    return pred


def decode_detect(pred, conf_thr, iou_thr, input_size, lb, src_w, src_h,
                  classes_filter=None):
    """Decode YOLO11 detect head → (boxes_xyxy, scores, classes) in orig coords."""
    pred = _to_anchors_channels(pred)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    classes = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    mask = confs >= conf_thr
    if classes_filter is not None:
        mask &= np.isin(classes, list(classes_filter))
    if not mask.any():
        return (np.empty((0, 4), np.float32), np.empty(0, np.float32),
                np.empty(0, np.int32))

    boxes = xywh2xyxy(boxes_xywh[mask] * float(input_size))  # normalized → input px
    confs = confs[mask]
    classes = classes[mask].astype(np.int32)
    keep = nms_classwise(boxes, confs, classes, iou_thr)
    boxes = scale_boxes_back(boxes[keep], lb, src_w, src_h)
    return boxes, confs[keep], classes[keep]


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-9)


def decode_classify(pred):
    """Decode a classify head (1, ncls) → (top1_index, top1_conf).
    Applies softmax only if the output is not already a probability vector."""
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    if not (abs(p.sum() - 1.0) < 0.1 and p.max() <= 1.0 and p.min() >= 0.0):
        p = softmax(p)
    idx = int(p.argmax())
    return idx, float(p[idx])
