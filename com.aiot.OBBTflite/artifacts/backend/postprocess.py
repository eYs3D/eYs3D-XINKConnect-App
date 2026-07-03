"""
YOLO11 OBB (oriented bounding box) post-processing (pure numpy).

Our per-tensor INT8 TFLite OBB export (Ultralytics SavedModel export + strict
per-tensor requant — see tools/convert_to_tflite.py): single output
(1, 4+nc+1, N) where
  [0:4]   box xywh NORMALIZED to [0,1] (relative to model input size)
  [4:4+nc] class scores (already activated)
  [-1]    rotation angle in radians (NOT normalized — already ~[-π/2, π/2],
          same magnitude as the [0,1] box so it shares the per-tensor scale fine)

Box is normalized so box and class scores share one per-tensor output scale
without box collapsing the scores onto the zero-point (see DetectTflite
postprocess). The decoder multiplies box (cx,cy,w,h) by input_size.

Rotated NMS uses ProbIoU (the Gaussian-Bhattacharyya overlap Ultralytics uses),
ported to numpy so it is unit-testable without cv2 / torch / tflite.
"""

import numpy as np


def _to_anchors_channels(pred):
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    return pred


def _covariance(obb):
    """obb (N,5) xywhr → (a, b, c) covariance components."""
    a = obb[:, 2] ** 2 / 12.0
    b = obb[:, 3] ** 2 / 12.0
    angle = obb[:, 4]
    cos = np.cos(angle)
    sin = np.sin(angle)
    cos2, sin2 = cos ** 2, sin ** 2
    return (a * cos2 + b * sin2,
            a * sin2 + b * cos2,
            (a - b) * cos * sin)


def probiou(obb1, obb2, eps=1e-7):
    """ProbIoU between two equal-length sets of xywhr boxes → (N,)."""
    x1, y1 = obb1[:, 0], obb1[:, 1]
    x2, y2 = obb2[:, 0], obb2[:, 1]
    a1, b1, c1 = _covariance(obb1)
    a2, b2, c2 = _covariance(obb2)

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5
    t3 = np.log(
        denom / (4.0 * np.sqrt((a1 * b1 - c1 ** 2).clip(0) *
                               (a2 * b2 - c2 ** 2).clip(0)) + eps) + eps
    ) * 0.5
    bd = (t1 + t2 + t3).clip(eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1.0 - hd


def nms_rotated(obbs, scores, iou_thr):
    """Greedy NMS over xywhr boxes using ProbIoU. Returns kept indices."""
    if len(obbs) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = probiou(np.repeat(obbs[i][None], len(rest), axis=0), obbs[rest])
        order = rest[ious <= iou_thr]
    return keep


def scale_obb_back(obbs, lb, src_w, src_h):
    """Map xywhr boxes from letterboxed pixels back to original image coords."""
    out = obbs.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - lb["pad_x"]) / lb["scale"]
    out[:, 1] = (out[:, 1] - lb["pad_y"]) / lb["scale"]
    out[:, 2] /= lb["scale"]
    out[:, 3] /= lb["scale"]
    return out


def decode_obb(pred, conf_thr, iou_thr, input_size, lb, src_w, src_h):
    """Decode YOLO11 OBB head → (obbs_xywhr, scores, classes).
    obbs are (M,5) cx,cy,w,h,angle(rad) in ORIGINAL image coords."""
    pred = _to_anchors_channels(pred)
    boxes_xywh = pred[:, :4]
    angle = pred[:, -1]
    cls_scores = pred[:, 4:-1]
    classes = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    mask = confs >= conf_thr
    if not mask.any():
        return (np.empty((0, 5), np.float32), np.empty(0, np.float32),
                np.empty(0, np.int32))

    xywh = boxes_xywh[mask] * float(input_size)  # normalized → input px (angle stays rad)
    obbs = np.concatenate([xywh, angle[mask, None]], axis=1).astype(np.float32)
    confs = confs[mask]
    classes = classes[mask].astype(np.int32)

    keep = nms_rotated(obbs, confs, iou_thr)
    obbs = scale_obb_back(obbs[keep], lb, src_w, src_h)
    return obbs, confs[keep], classes[keep]
