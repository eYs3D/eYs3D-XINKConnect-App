"""MiDaS TFLite wrapper for com.aiot.EYS3DDepthFusion.

Lifted from com.aiot.DepthTflite's main.py unchanged in behaviour. Each app in
this collection carries its own copy of the inference plumbing (tflite_backend.py
is byte-identical across all of them) so that any single component can be
deployed on its own — see the family README.
"""
import time

import numpy as np

import depth_postprocess as dp
import tflite_backend as tb


class DepthEstimator:
    """MiDaS TFLite wrapper handling both float and quantized models.

    Input/output dtype is read from the model rather than assumed, so the same
    code runs the INT8 model (NPU target) and the float32/float16 models (useful
    for A/B on x86) with no config change.
    """

    def __init__(self, model_path, delegate_path=None, input_size=256):
        self.interp, self.backend = tb.make_interpreter(
            model_path, delegate_path=delegate_path or None)
        tb.log_model_details(self.interp, self.backend, model_path)

        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        shape = list(self.inp["shape"])
        # onnx2tf converts NCHW->NHWC, so a 4-D input is [1,H,W,3]. Trust the
        # model's own H rather than config, so a mismatched input_size can't
        # silently produce a wrong-size feed.
        if len(shape) == 4 and shape[3] == 3:
            self.input_size = int(shape[1])
        else:
            self.input_size = int(input_size)
        if self.input_size != int(input_size):
            print(f"[depth] config midas_input_size={input_size} overridden by "
                  f"model input {self.input_size}", flush=True)

        self.last_ms = 0.0

    def infer(self, frame_bgr):
        """BGR frame -> float32 (H,W) inverse relative depth at model resolution."""
        blob = dp.preprocess(frame_bgr, self.input_size)

        dtype = self.inp["dtype"]
        if np.issubdtype(dtype, np.integer):
            blob = tb.quantize(blob, self.inp["quantization"], dtype)
        else:
            blob = blob.astype(dtype)

        t0 = time.perf_counter()
        self.interp.set_tensor(self.inp["index"], blob)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        depth = tb.dequantize(raw, self.out["quantization"])
        return np.squeeze(depth).astype(np.float32)
