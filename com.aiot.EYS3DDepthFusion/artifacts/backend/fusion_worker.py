"""Fusion thread: paired (rgb, depth) -> MiDaS -> fuse -> display frame.

Runs off the SDK callback threads on purpose. MiDaS inference takes tens of
milliseconds; doing it inside a callback would stall the C++ frame producer and
drop capture frames rather than display frames.

Every failure path here degrades to the raw SDK heatmap, which is exactly what
com.aiot.EYS3DCamera would have shown. Fusion is an enhancement layered on a
working baseline, so when it cannot run the baseline must still stream.
"""
import queue
import threading
import time

import numpy as np

import depth_fusion as df
import depth_postprocess as dp


# This component's config vocabulary -> the mode names depth_postprocess.compose
# actually understands. compose() is shared verbatim with com.aiot.DepthTflite,
# where the depth-only mode is called "depth"; here the same thing is called
# "fused". compose() returns SIDE-BY-SIDE for every name it does not recognise,
# so an unmapped mode does not fail — it silently renders the wrong layout.
# That is exactly what "fused" did before this table existed.
_COMPOSE_MODE = {
    "fused": "depth",
    "side_by_side": "side_by_side",
    "overlay": "overlay",
}

# Modes handled before compose() is ever reached, needing no inference.
_DIRECT_MODES = ("rgb", "raw_depth")

# Needs fusion but not compose(): both panels are depth, so there is no RGB
# frame involved and compose()'s rgb-on-the-left layout does not apply.
_COMPARE_MODE = "compare"

VALID_DISPLAY_MODES = tuple(_COMPOSE_MODE) + _DIRECT_MODES + (_COMPARE_MODE,)


def _resolve_colormap(name):
    """Colormap by name, adding "eys3d" on top of depth_postprocess's set.

    "eys3d" reproduces the SDK's own palette, so `fused` and `raw_depth` look
    like the same instrument rather than two different tools.
    """
    if str(name).lower() == "eys3d":
        return df.eys3d_palette()
    return dp.resolve_colormap(name)


class FusionThread(threading.Thread):
    def __init__(self, in_queue, last_frame_holder, estimator, fuser,
                 display_mode="fused", colormap="eys3d", overlay_alpha=0.5,
                 show_hud=True, midas_every_n=1,
                 display_min_mm=200.0, display_max_mm=1000.0, log=print):
        super().__init__(daemon=True, name="FusionThread")
        self._q = in_queue
        self._holder = last_frame_holder
        self._estimator = estimator
        self._fuser = fuser
        self._mode = str(display_mode)
        if self._mode not in VALID_DISPLAY_MODES:
            # Loud, because the failure mode is a picture that looks plausible
            # but is not what was asked for.
            log(f"[fusion] unknown display_mode {self._mode!r}; "
                f"falling back to 'fused'. Valid: {list(VALID_DISPLAY_MODES)}")
            self._mode = "fused"
        self._cmap = _resolve_colormap(colormap)
        self._disp_min = float(display_min_mm)
        self._disp_max = float(display_max_mm)
        self._alpha = float(overlay_alpha)
        self._show_hud = bool(show_hud)
        self._every_n = max(1, int(midas_every_n))
        self._log = log

        self._running = True
        self._frame_no = 0
        self._last_midas = None
        self._fps = 0.0
        self._last_ts = None
        self._warned_no_zd = False
        self._last_info = {}

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                rgb, z_mm, heat = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._holder.set(self._process(rgb, z_mm, heat))
            except Exception as exc:
                # A bad frame must not kill the stream; fall back to the raw
                # heatmap and keep going.
                self._log(f"[fusion] frame failed ({exc}); showing raw depth")
                self._holder.set(heat)
            self._tick_fps()

    def _process(self, rgb, z_mm, heat):
        self._frame_no += 1

        if self._mode == "rgb":
            return self._hud(rgb.copy(), extra=["mode=rgb"])
        if self._mode == "raw_depth" or self._estimator is None or z_mm is None:
            if z_mm is None and not self._warned_no_zd:
                self._log("[fusion] depth ZD values unavailable — raw heatmap only")
                self._warned_no_zd = True
            return self._hud(heat.copy(), extra=["mode=raw_depth"])

        if self._frame_no % self._every_n == 0 or self._last_midas is None:
            self._last_midas = self._estimator.infer(rgb)
        fused, info = self._fuser.fuse(z_mm, self._last_midas)
        self._last_info = info

        depth_color = df.colorize_metric(
            fused, self._disp_min, self._disp_max, self._cmap)

        if self._mode == _COMPARE_MODE:
            # Both panels are colorized HERE, through the same palette and
            # range, so the only visible difference is the filled holes. Using
            # the SDK's own heatmap on the left would look more "authentic" but
            # confound the comparison: it blacks out past 1000mm where this
            # component clips to violet, so the far field would differ for
            # reasons that have nothing to do with fusion.
            raw_color = df.colorize_metric(
                z_mm, self._disp_min, self._disp_max, self._cmap)
            return self._label_panels(np.hstack([raw_color, depth_color]))

        out = dp.compose(rgb, depth_color, mode=_COMPOSE_MODE[self._mode],
                         alpha=self._alpha)
        return self._hud(out)

    def _label_panels(self, out):
        """Mark which half is which — a side-by-side of two depth maps is
        otherwise ambiguous, and reading it backwards inverts the conclusion."""
        frame = self._hud(out)
        if self._show_hud:
            half = out.shape[1] // 2
            y = out.shape[0] - 10
            dp.draw_hud(frame, ["raw"], origin=(8, y))
            dp.draw_hud(frame, ["fused"], origin=(half + 8, y))
        return frame

    def _tick_fps(self):
        now = time.monotonic()
        if self._last_ts is not None:
            dt = now - self._last_ts
            if dt > 0:
                # EMA so the HUD reads steadily instead of jittering per frame.
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last_ts = now

    def _hud(self, frame, extra=None):
        if not self._show_hud:
            return frame
        lines = [f"{self._fps:4.1f} fps  mode={self._mode}"]
        if self._estimator is not None:
            lines.append(f"midas {self._estimator.backend} {self._estimator.last_ms:.0f}ms")
        info = self._last_info
        if info:
            if info.get("fused"):
                lines.append(f"fused  valid {info['valid_ratio']*100:.0f}%  "
                             f"filled {info['filled_ratio']*100:.0f}%")
            else:
                lines.append(f"passthrough: {info.get('reason', '')}")
        if extra:
            lines.extend(extra)
        return dp.draw_hud(frame, lines)
