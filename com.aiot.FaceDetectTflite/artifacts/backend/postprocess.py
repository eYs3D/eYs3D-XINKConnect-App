"""
SCRFD post-processing (pure numpy, shared by the ONNX and TFLite backends).

Extracted from main.py so the decode math is importable and unit-testable
without pulling in cv2 / av / tflite_runtime.

SCRFD outputs 9 tensors (with KPS) or 6 tensors (without KPS), one group per
FPN stride (8, 16, 32). Each group: score[N,1], bbox[N,4], kps[N,10].
N = (input_size / stride)^2 * 2   (2 anchors per grid cell).

Bounding-box decode (distance-from-center formulation):
  x1 = cx - Δ0·s,  y1 = cy - Δ1·s
  x2 = cx + Δ2·s,  y2 = cy + Δ3·s

Keypoint decode (5 × (dx, dy) interleaved):
  kp_x = cx + Δ[2j]·s,  kp_y = cy + Δ[2j+1]·s   for j in 0..4
"""

import numpy as np


def _nms(boxes, scores, iou_thr):
    """Greedy NMS, pure numpy. Returns list of kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
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
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def _classify_scrfd_outputs(outs):
    """
    Classify SCRFD output tensors by their last dimension.
    Returns (score_ii, bbox_ii, kps_ii) sorted DESC by anchor count.
    Works identically for both ONNX and TFLite output lists.
    """
    score_ii, bbox_ii, kps_ii = [], [], []
    for i, o in enumerate(outs):
        d = o.shape[-1]
        if d == 1:
            score_ii.append(i)
        elif d == 4:
            bbox_ii.append(i)
        elif d == 10:
            kps_ii.append(i)

    # Sort DESC by n_anchors so stride-8 (most anchors) comes first
    score_ii.sort(key=lambda i: outs[i].shape[-2], reverse=True)
    bbox_ii.sort(key=lambda i: outs[i].shape[-2], reverse=True)
    kps_ii.sort(key=lambda i: outs[i].shape[-2], reverse=True)
    return score_ii, bbox_ii, kps_ii


def _build_anchor_levels(score_ii, outs, input_size):
    """Pre-compute anchor center grids for each FPN stride level."""
    s = input_size
    levels = []
    for idx in score_ii:
        n = outs[idx].shape[-2]         # n = (s/stride)^2 * 2
        fs = int(round(np.sqrt(n / 2))) # feature-map side length
        stride = s // fs
        gy, gx = np.mgrid[0:fs, 0:fs]
        ac = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        ac = np.repeat(ac, 2, axis=0) * stride  # (n, 2)
        levels.append((stride, ac))
        print(f"[SCRFD]   stride={stride:2d}  anchors={n}", flush=True)
    return levels


def _decode_scrfd_outputs(outs, score_ii, bbox_ii, kps_ii, levels,
                          apply_sigmoid, threshold, nms_iou, img_w, img_h, input_size):
    """Shared SCRFD decode + NMS logic for both ONNX and TFLite."""
    s = input_size
    sx, sy = img_w / s, img_h / s
    all_boxes, all_scores, all_kps = [], [], []
    has_kps = kps_ii is not None and len(kps_ii) > 0

    for lv, (stride, ac) in enumerate(levels):
        raw_sc = outs[score_ii[lv]].reshape(-1, 1)[:, 0]
        if apply_sigmoid:
            raw_sc = 1.0 / (1.0 + np.exp(-raw_sc))

        mask = raw_sc >= threshold
        if not mask.any():
            continue

        sc  = raw_sc[mask]
        bd  = outs[bbox_ii[lv]].reshape(-1, 4)[mask]
        ctr = ac[mask]

        boxes = np.empty((len(sc), 4), np.float32)
        boxes[:, 0] = (ctr[:, 0] - bd[:, 0] * stride) * sx
        boxes[:, 1] = (ctr[:, 1] - bd[:, 1] * stride) * sy
        boxes[:, 2] = (ctr[:, 0] + bd[:, 2] * stride) * sx
        boxes[:, 3] = (ctr[:, 1] + bd[:, 3] * stride) * sy
        np.clip(boxes[:, 0], 0, img_w, out=boxes[:, 0])
        np.clip(boxes[:, 1], 0, img_h, out=boxes[:, 1])
        np.clip(boxes[:, 2], 0, img_w, out=boxes[:, 2])
        np.clip(boxes[:, 3], 0, img_h, out=boxes[:, 3])

        all_boxes.append(boxes)
        all_scores.append(sc)

        if has_kps:
            kd = outs[kps_ii[lv]].reshape(-1, 10)[mask]
            kx = (ctr[:, 0:1] + kd[:, 0::2] * stride) * sx
            ky = (ctr[:, 1:2] + kd[:, 1::2] * stride) * sy
            all_kps.append(np.stack([kx, ky], axis=2))

    if not all_boxes:
        empty_kps = np.empty((0, 5, 2), np.float32) if has_kps else None
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), empty_kps

    boxes  = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    kps    = np.vstack(all_kps) if has_kps and all_kps else None

    keep = _nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep], kps[keep] if kps is not None else None
