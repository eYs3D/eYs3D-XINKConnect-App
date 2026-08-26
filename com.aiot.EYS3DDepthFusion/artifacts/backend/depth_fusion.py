"""Fuse eYs3D metric stereo depth with MiDaS monocular depth.

The two sources fail in opposite ways, which is what makes fusing them worth
doing at all:

  * Stereo (eYs3D)  — metric millimetres, but holes wherever matching fails
                      (textureless walls, specular highlights, occlusion) and
                      per-frame flicker because each frame is matched
                      independently.
  * MiDaS           — dense and stable, but *affine-invariant inverse* depth:
                      the numbers carry no metric meaning, only ordering.

So MiDaS can never be trusted for magnitude, and stereo can never be trusted
for coverage. This module uses stereo to pin MiDaS to metric scale, then lets
MiDaS supply only the pixels stereo could not measure.

ALL FITTING HAPPENS IN INVERSE DEPTH (1/mm), for two reasons:

  1. MiDaS is linear there and nowhere else. It predicts `d ≈ s·(1/Z) + t`, so
     inverse space is where a 2-parameter fit is actually correct. Fitting
     against Z directly would be fitting a hyperbola with a line.
  2. Error behaves. A fixed inverse-depth error is a small absolute error up
     close and a large one far away — which is exactly how stereo uncertainty
     really grows. Doing the correction in millimetres instead would let
     far-field noise dominate the whole fit.

Nothing here imports TFLite or the eYs3D SDK, so the whole algorithm is
unit-testable off-device — see tests/greengrass/com.aiot.EYS3DDepthFusion/.
"""

import cv2
import numpy as np

# Sub-sampling stride for the fit. At 1280x720 stride 4 still leaves ~57k
# samples for a 2-parameter fit, which is over-determined by four orders of
# magnitude; going finer costs time and buys nothing.
DEFAULT_SUBSAMPLE = 4

# Box-filter width for the residual correction field. Must be wide enough to
# span a typical hole so the correction is interpolated ACROSS it from the
# valid rim rather than being dominated by a couple of edge pixels.
DEFAULT_CORRECTION_KERNEL = 101

# Beyond this distance a filled pixel is not trustworthy enough to publish.
# Measured on the ecv board (PID 0x181, mode 1) against held-out real readings:
#
#     true depth       median fill error     within 25%
#     340-  488 mm            8 mm             99.6%
#     488-  843 mm           18 mm             98.5%
#     843- 1175 mm        35-47 mm           95-99%
#    1175- 2871 mm           98 mm             88.0%
#    2871+      mm         1107 mm             39.2%
#
# The cliff is inherent, not a defect: the alignment is affine in INVERSE depth,
# so a fixed inverse-depth error is millimetres up close and metres far away.
# Past this range a hole is more honest than a confident metre-wrong number, so
# far fills are dropped rather than published. 0 disables the cap.
DEFAULT_MAX_FILL_MM = 2500.0


def valid_mask(z_mm, z_min, z_max):
    """Pixels carrying a real stereo reading.

    Holes are 0, but a plain `z > 0` test is not enough: the sensor also emits
    values outside its own working range, which are noise rather than
    measurements. `Device.get_z_range()` supplies the real bounds.
    """
    return (z_mm >= z_min) & (z_mm <= z_max)


def fit_scale_shift(midas, inv_z, trim_sigma=2.5, trim_rounds=2):
    """Least-squares fit of `inv_z ≈ s·midas + t` over paired 1-D samples.

    Returns (s, t) or None when the fit is degenerate or non-physical.

    Trimmed rather than RANSAC-fitted, deliberately. Stereo depth carries
    outliers — flying pixels at depth discontinuities, mismatches on repetitive
    texture — and plain least squares chases them. The textbook answer is
    RANSAC, but RANSAC is randomised: identical input yields slightly different
    (s, t) on each call, and that jitter would show up in the output as exactly
    the flicker this component exists to remove. Trimming is deterministic, so
    a static scene produces a static fit.
    """
    d = np.asarray(midas, dtype=np.float64).ravel()
    y = np.asarray(inv_z, dtype=np.float64).ravel()
    if d.size < 16:
        return None

    keep = np.ones(d.size, dtype=bool)
    s = t = 0.0
    for _ in range(max(1, int(trim_rounds) + 1)):
        dk, yk = d[keep], y[keep]
        if dk.size < 16:
            return None
        var = float(dk.var())
        if var < 1e-18:
            return None  # MiDaS output is flat — nothing to align against
        s = float(((dk - dk.mean()) * (yk - yk.mean())).mean() / var)
        t = float(yk.mean() - s * dk.mean())

        resid = s * d + t - y
        sigma = float(resid[keep].std())
        if sigma < 1e-18:
            break
        keep = np.abs(resid) <= trim_sigma * sigma

    # MiDaS emits INVERSE depth, so larger output means closer, the same
    # direction as 1/Z. A non-positive slope means the fit locked onto noise
    # and inverting it would render the scene inside-out.
    if not np.isfinite(s) or not np.isfinite(t) or s <= 0.0:
        return None
    return s, t


class ScaleShiftTracker:
    """EMA over the fitted (s, t), with rejection of bad frames.

    True scene scale does not jump between consecutive frames, so smoothing the
    two fit parameters removes most inter-frame flicker for the price of two
    multiply-adds — far cheaper than filtering the depth map itself, and with
    no motion ghosting, because it never mixes pixels across time.

    A rejected frame keeps the previous parameters rather than falling back to
    an unaligned map: stale scale is a much smaller error than wrong scale.
    """

    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.s = None
        self.t = None

    def update(self, fit):
        if fit is None:
            return (self.s, self.t) if self.s is not None else None
        s, t = fit
        if self.s is None:
            self.s, self.t = float(s), float(t)
        else:
            a = self.alpha
            self.s = a * float(s) + (1.0 - a) * self.s
            self.t = a * float(t) + (1.0 - a) * self.t
        return self.s, self.t

    def reset(self):
        self.s = None
        self.t = None


def _masked_blur(values, mask_f, kernel):
    """Box-blur `values` averaging ONLY where mask_f is 1.

    Normalising by the blurred mask is what makes this work over holes: a
    plain blur would average the zeros stored in hole pixels back in and drag
    the correction toward zero precisely where correction matters most.
    """
    k = (int(kernel) | 1, int(kernel) | 1)  # box filters need odd extents
    num = cv2.blur(values * mask_f, k)
    den = cv2.blur(mask_f, k)
    out = np.zeros_like(num)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out, ok


def eys3d_palette():
    """256-entry BGR LUT reproducing the eYs3D SDK's own depth colorizer.

    Derived from a real capture rather than from SDK source: sampling the SDK's
    heatmap against the metric depth of the SAME frame gives a plain
    full-saturation hue sweep, linear in depth,

        hue_degrees = (z_mm - 200) / 800 * 240

    i.e. red at 200mm through violet at 1000mm. Measured on the ecv board
    (PID 0x181): 400mm->60 deg, 500->90, 600->120, 700->152, 800->188, 950->232.
    The SDK paints everything beyond 1000mm black; this LUT does not, because
    black already means "no reading" here and reusing it for "far" would make
    filled pixels indistinguishable from holes.

    Built REVERSED because colorize_metric indexes the LUT with
    (1 - normalized_depth) * 255, so entry 255 is the nearest pixel.
    """
    idx = np.arange(256, dtype=np.float32)
    hsv = np.zeros((256, 1, 3), dtype=np.uint8)
    # OpenCV packs hue into 0..179 for a 0..359 degree range, hence the halving.
    hsv[:, 0, 0] = ((255.0 - idx) / 255.0 * 240.0 / 2.0).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def colorize_metric(z_mm, z_min, z_max, colormap):
    """Metric millimetres -> BGR heatmap over a FIXED range, near = bright.

    Fixed-range, unlike the per-frame normalization MiDaS-only display requires
    (see depth_postprocess.DepthNormalizer). Because these values are real
    distances, the same colour means the same distance in every frame — so a
    static scene renders a static image, and the display stops pulsing as
    objects enter and leave. That stability is half the point of fusing.

    Watch the sign: normalized metric depth runs 0=near to 1=far, the OPPOSITE
    of MiDaS's inverse depth. Near-bright therefore needs the flip here, where
    for MiDaS it is the pass-through case.

    The range is the DISPLAY range, not the validity range, and picking it too
    wide silently destroys contrast: spanning 300..8000mm put a scene that
    actually occupies 350..1200mm into LUT entries 231..252, so everything
    rendered as one shade of red. Match it to where the data is.

    `colormap` may be a cv2 constant, a 256x1x3 BGR LUT (see eys3d_palette),
    or None for grayscale.
    """
    span = max(1e-6, float(z_max) - float(z_min))
    norm = np.clip((np.asarray(z_mm, dtype=np.float32) - float(z_min)) / span, 0.0, 1.0)
    u8 = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
    if colormap is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    out = cv2.applyColorMap(u8, colormap)
    # Unmeasured and unfillable pixels are 0 mm, which would otherwise render
    # as the "nearest" colour and read as an object right in front of the lens.
    out[np.asarray(z_mm) <= 0] = 0
    return out


class DepthFuser:
    """Hole-filling fusion of stereo metric depth with monocular MiDaS depth.

    Call `fuse(z_mm, midas)` per frame. Output is metric millimetres, float32,
    with 0 where no estimate could be produced at all.
    """

    def __init__(self, z_min, z_max, ema_alpha=0.3, trim_sigma=2.5,
                 trim_rounds=2, min_valid_ratio=0.05,
                 correction_kernel=DEFAULT_CORRECTION_KERNEL,
                 subsample=DEFAULT_SUBSAMPLE, local_correction=True,
                 max_fill_mm=DEFAULT_MAX_FILL_MM):
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.trim_sigma = float(trim_sigma)
        self.trim_rounds = int(trim_rounds)
        self.min_valid_ratio = float(min_valid_ratio)
        self.correction_kernel = int(correction_kernel)
        self.subsample = max(1, int(subsample))
        self.local_correction = bool(local_correction)
        self.max_fill_mm = float(max_fill_mm)
        self.tracker = ScaleShiftTracker(alpha=ema_alpha)

    def fuse(self, z_mm, midas):
        """(stereo mm, MiDaS inverse depth) -> (fused mm float32, info dict).

        `midas` may be any resolution; it is resized to the stereo frame's
        geometry. Fusion is per-pixel, so it assumes the depth map and the
        image MiDaS ran on share a viewpoint — see README "Registration".

        On any failure the ORIGINAL stereo depth is returned unchanged. Fusion
        must never make the output worse than the baseline it replaces, so
        every early exit degrades to plain passthrough rather than to a
        best-effort guess.
        """
        z = np.asarray(z_mm)
        h, w = z.shape[:2]
        zf = z.astype(np.float32)
        info = {"fused": False, "reason": "", "s": None, "t": None,
                "valid_ratio": 0.0, "filled_ratio": 0.0,
                "dropped_far_ratio": 0.0}

        m = valid_mask(z, self.z_min, self.z_max)
        valid_ratio = float(m.mean())
        info["valid_ratio"] = valid_ratio
        if valid_ratio < self.min_valid_ratio:
            # Too little metric evidence to pin down scale. Anything produced
            # here would be MiDaS's guess wearing a metric costume.
            info["reason"] = f"insufficient valid stereo ({valid_ratio:.3f})"
            return zf, info

        d = np.asarray(midas, dtype=np.float32)
        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        inv_z = np.zeros((h, w), dtype=np.float32)
        np.divide(1.0, zf, out=inv_z, where=m)

        step = self.subsample
        sm, sd, si = m[::step, ::step], d[::step, ::step], inv_z[::step, ::step]
        fit = fit_scale_shift(sd[sm], si[sm], self.trim_sigma, self.trim_rounds)
        if fit is None and self.tracker.s is None:
            info["reason"] = "scale/shift fit failed and no previous fit"
            return zf, info

        params = self.tracker.update(fit)
        if params is None or params[0] is None:
            info["reason"] = "no usable scale/shift"
            return zf, info
        s, t = params
        info["s"], info["t"] = s, t
        if fit is None:
            info["reason"] = "fit rejected; reused previous parameters"

        inv_hat = s * d + t

        if self.local_correction:
            mask_f = m.astype(np.float32)
            resid = np.zeros((h, w), dtype=np.float32)
            np.subtract(inv_z, inv_hat, out=resid, where=m)
            corr, corr_ok = _masked_blur(resid, mask_f, self.correction_kernel)
            # Where no valid pixel lies within the kernel there is nothing to
            # interpolate from, so leave the global fit uncorrected there.
            inv_hat = inv_hat + np.where(corr_ok, corr, 0.0).astype(np.float32)

        # An inverse depth at or below zero is a point at or beyond infinity —
        # unrepresentable, so those pixels stay holes rather than becoming
        # absurd distances.
        fillable = (~m) & (inv_hat > 0)
        est = np.zeros((h, w), dtype=np.float32)
        np.divide(1.0, inv_hat, out=est, where=fillable)

        if self.max_fill_mm > 0:
            too_far = fillable & (est > self.max_fill_mm)
            fillable = fillable & ~too_far
            info["dropped_far_ratio"] = float(too_far.mean())

        out = zf.copy()
        out[fillable] = np.clip(est[fillable], self.z_min, self.z_max)

        info["fused"] = True
        info["filled_ratio"] = float(fillable.mean())
        return out, info
