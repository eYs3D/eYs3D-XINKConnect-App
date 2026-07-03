"""
YOLO11 pose post-processing (pure numpy).

Our per-tensor INT8 TFLite pose export (Ultralytics SavedModel export + strict
per-tensor requant — see tools/convert_to_tflite.py): single output
(1, 4+1+51, N) where
  [0:4]  box xywh NORMALIZED to [0,1] (relative to model input size)
  [4]    person confidence (already activated)
  [5:56] 17 keypoints × (x, y, conf) — x,y NORMALIZED to [0,1], conf activated

Box and keypoint x,y are normalized so that box, scores and keypoints all share
one per-tensor output scale without the box range collapsing the class/conf
scores onto the quantization zero-point (see DetectTflite postprocess). The
decoder multiplies box AND keypoint x,y by input_size to recover input pixels.

Numpy-only so it is unit-testable without cv2 / tflite.
"""

import numpy as np

N_KPTS = 17


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


def scale_boxes_back(boxes, lb, src_w, src_h):
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - lb["pad_x"]) / lb["scale"]
    out[:, [1, 3]] = (out[:, [1, 3]] - lb["pad_y"]) / lb["scale"]
    np.clip(out[:, [0, 2]], 0, src_w, out=out[:, [0, 2]])
    np.clip(out[:, [1, 3]], 0, src_h, out=out[:, [1, 3]])
    return out


def scale_points_back(pts, lb, src_w, src_h):
    """pts shape (..., 2) in letterboxed pixels → original image coords."""
    out = pts.astype(np.float32).copy()
    out[..., 0] = (out[..., 0] - lb["pad_x"]) / lb["scale"]
    out[..., 1] = (out[..., 1] - lb["pad_y"]) / lb["scale"]
    np.clip(out[..., 0], 0, src_w, out=out[..., 0])
    np.clip(out[..., 1], 0, src_h, out=out[..., 1])
    return out


def _to_anchors_channels(pred):
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    return pred


def decode_pose(pred, conf_thr, iou_thr, input_size, lb, src_w, src_h):
    """Decode YOLO11 pose head → (boxes_xyxy, scores, keypoints).

    keypoints shape (M, 17, 3): x,y in ORIGINAL image coords, conf in [0,1].
    """
    pred = _to_anchors_channels(pred)
    boxes_xywh = pred[:, :4]
    confs = pred[:, 4]
    kpts = pred[:, 5:5 + N_KPTS * 3]

    mask = confs >= conf_thr
    if not mask.any():
        return (np.empty((0, 4), np.float32), np.empty(0, np.float32),
                np.empty((0, N_KPTS, 3), np.float32))

    boxes = xywh2xyxy(boxes_xywh[mask] * float(input_size))  # normalized → input px
    confs = confs[mask]
    kpts = kpts[mask].reshape(-1, N_KPTS, 3).astype(np.float32)
    kpts[..., :2] *= float(input_size)  # normalized → input px (conf channel left as-is)

    keep = nms(boxes, confs, iou_thr)
    boxes = scale_boxes_back(boxes[keep], lb, src_w, src_h)
    kpts = kpts[keep]
    kpts[..., :2] = scale_points_back(kpts[..., :2], lb, src_w, src_h)
    return boxes, confs[keep], kpts
