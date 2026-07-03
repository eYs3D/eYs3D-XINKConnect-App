"""
YOLO11/YOLOv8 segmentation post-processing (pure numpy).

Our per-tensor INT8 TFLite seg export (Ultralytics SavedModel export + strict
per-tensor requant — see tools/convert_to_tflite.py) has TWO outputs:
  detection head : (1, 4+nc+nm, N)  — boxes(xywh NORMALIZED [0,1]) + class scores + mask coeffs
  mask prototypes: (1, mh, mw, nm)  (or (1, nm, mh, mw))

Box is normalized so box, class scores and mask coeffs share one per-tensor
output scale without box collapsing the scores onto the zero-point (see
DetectTflite postprocess). The decoder multiplies box by input_size; mask
coefficients and prototypes are left untouched.

decode_seg returns low-resolution per-instance masks (M, mh, mw); the caller
resizes/crops/overlays them with cv2. Numpy-only → unit-testable.
"""

import numpy as np

N_MASK = 32


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


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


def crop_mask(masks, boxes):
    """Zero each mask outside its own box (boxes in MASK-grid pixel coords).

    YOLO-seg masks are sigmoid(coeffs @ protos), a linear combo that ripples
    across the WHOLE grid — high values land far outside the object. Ultralytics
    always crops each instance mask to its box before thresholding; without it
    the background lights up everywhere (e.g. a person mask smeared over the sky).
    masks: (M, mh, mw); boxes: (M, 4) x1,y1,x2,y2 in mask-grid pixels."""
    _, mh, mw = masks.shape
    x1 = boxes[:, 0:1][:, :, None]
    y1 = boxes[:, 1:2][:, :, None]
    x2 = boxes[:, 2:3][:, :, None]
    y2 = boxes[:, 3:4][:, :, None]
    c = np.arange(mw, dtype=np.float32)[None, None, :]
    r = np.arange(mh, dtype=np.float32)[None, :, None]
    keep = (c >= x1) & (c < x2) & (r >= y1) & (r < y2)
    return masks * keep


def _protos_hwc(protos, n_mask):
    """Normalize prototypes to (mh, mw, n_mask)."""
    protos = np.asarray(protos, dtype=np.float32)
    if protos.ndim == 4:
        protos = protos[0]
    if protos.shape[0] == n_mask:           # (nm, mh, mw) → (mh, mw, nm)
        protos = np.transpose(protos, (1, 2, 0))
    return protos


def decode_seg(pred_det, protos, conf_thr, iou_thr, input_size, lb,
               src_w, src_h, n_mask=N_MASK):
    """Decode seg head + prototypes → (boxes_xyxy, scores, classes, masks).

    boxes are ORIGINAL image coords; masks are (M, mh, mw) sigmoid maps in [0,1]
    spanning the (letterboxed) model input — caller maps them to the frame.
    """
    pred = _to_anchors_channels(pred_det)
    nc = pred.shape[1] - 4 - n_mask
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:4 + nc]
    coeffs = pred[:, 4 + nc:4 + nc + n_mask]
    classes = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    mh = mw = 0
    protos_hwc = _protos_hwc(protos, n_mask)
    mh, mw, _ = protos_hwc.shape

    mask = confs >= conf_thr
    if not mask.any():
        return (np.empty((0, 4), np.float32), np.empty(0, np.float32),
                np.empty(0, np.int32), np.empty((0, mh, mw), np.float32))

    boxes = xywh2xyxy(boxes_xywh[mask] * float(input_size))  # normalized → input px
    confs = confs[mask]
    classes = classes[mask].astype(np.int32)
    sel_coeffs = coeffs[mask]

    keep = nms_classwise(boxes, confs, classes, iou_thr)
    sel_coeffs = sel_coeffs[keep]

    proto_flat = protos_hwc.reshape(-1, n_mask)            # (mh*mw, nm)
    masks = sigmoid(sel_coeffs @ proto_flat.T)             # (M, mh*mw)
    masks = masks.reshape(-1, mh, mw)

    # Crop each mask to its box (in mask-grid coords) so background ripples are
    # zeroed — boxes are still in input-px here, scale to the mh×mw grid first.
    kept_boxes = boxes[keep]
    grid_boxes = kept_boxes.copy()
    grid_boxes[:, [0, 2]] *= mw / float(input_size)
    grid_boxes[:, [1, 3]] *= mh / float(input_size)
    masks = crop_mask(masks, grid_boxes)

    boxes = scale_boxes_back(kept_boxes, lb, src_w, src_h)
    return boxes, confs[keep], classes[keep], masks
