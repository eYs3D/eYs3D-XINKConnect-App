# com.aiot.EYS3DDepthFusion

Streams eYs3D stereo depth with its holes filled by MiDaS monocular depth, as
length-prefixed H264 over TCP.

The two depth sources fail in opposite ways. Stereo gives true millimetres but
drops out on textureless walls, specular surfaces and occlusions, and flickers
because every frame is matched independently. MiDaS is dense and stable but
returns *affine-invariant inverse* depth — ordering only, no metric meaning.
This component uses stereo to pin MiDaS to metric scale, then lets MiDaS supply
only the pixels stereo could not measure.

## Why this is a separate component

`com.aiot.EYS3DCamera` and `com.aiot.DepthTflite` stay minimal single-purpose
examples. Fusion cannot be bolted onto either of them from outside:

* It needs metric millimetres from `Frame.get_depth_ZD_value()`. All
  `EYS3DCamera` puts on the wire is the SDK's **colorized heatmap**, pushed
  through a lossy H264 encoder — millimetres are unrecoverable downstream, and
  holes become indistinguishable from genuinely-distant pixels.
* The camera is an **exclusive** device, so `DepthTflite` cannot open it
  alongside `EYS3DCamera` to read depth itself.

So fusion has to run in the process that owns the camera. This component is a
copy of `EYS3DCamera` with MiDaS and the fusion stage added. It is **mutually
exclusive** with `EYS3DCamera` — deploy one or the other, never both.

## Algorithm

All fitting happens in **inverse depth (1/mm)**, because that is the only space
where MiDaS is linear (`d ≈ s·(1/Z) + t`) and where error growth matches how
stereo uncertainty actually scales with distance.

1. **Validity mask** — from `Device.get_z_range()`, not just `z > 0`. The sensor
   emits out-of-range values that are noise, not measurements.
2. **Robust scale/shift fit** — least squares over valid pixels, then two
   trimming rounds at 2.5σ. Trimmed rather than RANSAC-fitted on purpose:
   RANSAC is randomised, so identical input gives slightly different parameters
   each frame, and that jitter would surface as exactly the flicker this
   component exists to remove.
3. **EMA of (s, t)** — true scene scale does not jump between frames. Smoothing
   two scalars kills most inter-frame flicker with no motion ghosting, because
   it never mixes pixels across time. A rejected fit reuses the previous
   parameters; stale scale beats wrong scale.
4. **Fill holes only** — measured pixels are always kept. MiDaS never overwrites
   a real reading.
5. **Residual-guided local correction** — a mask-normalised box blur of the
   fit residual, added back to the aligned map. This makes fills meet the
   measured data continuously at hole edges instead of stepping, and absorbs
   slowly-varying misalignment so the global fit need not be perfect.

Every failure path degrades to the raw SDK heatmap — the same picture
`EYS3DCamera` would have shown. Fusion is an enhancement over a working
baseline, so when it cannot run the baseline still streams.

## Display modes

| `display_mode` | Shows |
|---|---|
| `fused` | fused depth, colorized over a fixed metric range |
| **`compare`** | **raw depth \| fused — the mode for judging this component** |
| `raw_depth` | SDK heatmap only — the like-for-like baseline |
| `rgb` | color image only |
| `side_by_side` | RGB \| fused |
| `overlay` | fused blended onto RGB |

**`compare` is the one to look at.** Both panels are colorized here, through the
same palette and the same range, so the only visible difference between them is
the holes fusion filled. The panels are labelled `raw` and `fused` along the
bottom, because a side-by-side of two depth maps is otherwise ambiguous and
reading it backwards inverts the conclusion.

It deliberately does *not* put the SDK's own heatmap on the left. That would
look more authentic but confound the comparison: the SDK blacks out beyond
1000mm where this component clips to violet, so the far field would differ for
reasons unrelated to fusion. Use `raw_depth` when you want the SDK's actual
output.

## Colour

`colormap: "eys3d"` reproduces the SDK's own palette — a full-saturation hue
sweep, red near through violet far — so `fused` and `raw_depth` read as the same
instrument. It was derived by sampling the SDK's heatmap against the metric
depth of the same captured frame: `hue = (z_mm - 200) / 800 * 240`, matching to
within 6 degrees median. cv2 names (`jet`, `turbo`, `inferno`, ...) also work.

`display_min_mm` / `display_max_mm` are the DISPLAY range and are deliberately
separate from the `z_min_mm` / `z_max_mm` validity range. Validity decides what
counts as a reading; display decides where colour resolution is spent.
Conflating them is a real trap: colorizing over 300..8000mm put a scene actually
occupying 350..1200mm into a 24-degree slice of hue, rendering the whole image
one shade of red. The defaults (200..1000mm) match the SDK; raise
`display_max_mm` to see far fills in colour at the cost of near-field contrast.

The range is fixed rather than per-frame min/max. Because these are real
distances, one colour means one distance in every frame, so a static scene
renders a static image.

Set `display_mode: "raw_depth"` with `depth_filter_enabled: false` to see the
unassisted stereo depth.

## Measured on hardware

ecv board, module PID `0x181`, `mode_index 1`, 1280x720 color and depth.

**Frame pairing.** Color frames carry EVEN serial numbers and depth frames ODD,
so `color_sn - depth_sn == -1` for a matched pair (36 of 65 pairs; the rest are
nearest-time mismatches). The SDK's parity note is literally true. Requiring
exact serial equality would form **zero** pairs, which is why `max_serial_delta`
defaults to 1.

**Metric depth.** `get_depth_ZD_value()` IS populated in callback mode:
uint16, 921600 values, exactly one per pixel. Readings run 338mm upward with a
16383mm saturation ceiling (14-bit).

**Holes.** 63% of the depth frame at the device's default IR level. This is the
problem the component exists for — and IR is the cheapest lever on it (below).

**MiDaS throughput.** 226ms/frame via the VX delegate versus 1304ms on CPU — the
NPU works (5.8x), MiDaS-small is simply expensive. That is 4.4/sec against a
15fps stream, hence `midas_every_n: 4`.

**`get_z_range()` is useless here** — returns `{'Near': 0, 'Far': 0}`, so the
`z_min_mm`/`z_max_mm` config fallback is the normal path, not an edge case.

## Try IR before trusting fusion

The projector paints texture onto the textureless surfaces that cause most
dropouts, so it attacks the same holes fusion does — but from the sensor side,
producing real measurements. Measured on the board:

| `ir_value` | holes | valid pixels |
|---|---|---|
| 0 | 51.5% | 445,673 |
| 6 *(device default)* | 41.4% | 540,207 |
| 11 | 32.4% | 622,078 |
| **15** *(needs `ir_extended`)* | **29.4%** | **650,859** |

Set `ir_value: 15, ir_extended: true` and a third of the holes disappear before
fusion runs. Unset by default, because it drives an emitter on your hardware
and the useful level is scene-dependent — it does nothing beyond the
projector's throw, and reflective surfaces can bloom.

On the 429,063 pixels valid at both 6 and 15, noise drops from 1.97mm to
1.35mm. The 96,345 pixels IR newly recovers are grainier (16.6mm) — real
readings where there was nothing, but expect those regions to look noisier.
Beware measuring this as "temporal std over always-valid pixels": that set
changes with IR, so the aggregate moves the wrong way while every individual
pixel improves.

## Accuracy, and where it stops

Scored by hold-out on a real capture (hide pixels stereo did measure, fill them,
compare against what was hidden):

| true depth | median fill error | within 25% |
|---|---|---|
| 340–488 mm | 8 mm | 99.6% |
| 488–843 mm | 18 mm | 98.5% |
| 843–1175 mm | 35–47 mm | 95–99% |
| 1175–2871 mm | 98 mm | 88.0% |
| 2871 mm+ | **1107 mm** | **39.2%** |

The cliff is inherent, not a defect: alignment is affine in INVERSE depth, so a
fixed inverse-depth error is millimetres up close and metres far away.
`fusion_max_fill_mm` (default 2500) therefore drops far fills instead of
publishing them — a hole reads as "unknown", whereas a number reads as a
measurement.

Note the cap filters on the *estimate*, which is all that exists at runtime. A
distant pixel that MiDaS underestimates as near still gets published, so a
residual tail survives: on the capture, median error is 27mm and 95% land within
25%, but RMSE is 400mm because ~5% of pixels are badly wrong.

## Registration — still unverified

Fusion is **per-pixel**, so it assumes the depth map and the image MiDaS ran on
share a viewpoint. Color and depth are confirmed the same geometry on this
module, but that is not the same as being registered. If the SDK delivers depth
in the depth-sensor frame, a warp is required first and output will be subtly
misaligned at object boundaries. The near-field accuracy above suggests the
views are close to aligned, but it is not proof.

`main.py` warns when color and depth resolutions differ. Per `ModeConfig.db`,
17 of 20 modules use identical resolutions at `mode_index 1`; the exceptions
(8038, Taryn, 8029) are modules whose color stream is the raw stereo pair.

## Status

Algorithm validated against a real capture; the component itself has not yet
been run end-to-end on the board.
