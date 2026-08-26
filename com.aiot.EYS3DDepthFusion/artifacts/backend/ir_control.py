"""IR projector control for com.aiot.EYS3DDepthFusion.

The projector paints texture onto surfaces that have none, which is the direct
cause of most stereo dropouts — so it attacks the same holes fusion exists to
fill, but from the sensor side, producing REAL measurements instead of
estimates. Where IR and fusion can both reach a pixel, IR wins every time.

Measured on the ecv board (PID 0x181, mode 1, static scene, settled):

    IR    holes    valid px
     0    51.5%     445673
     6    41.4%     540207    <- device default (repeat run: 41.1%, 543258)
     8    36.7%     582839
    11    32.4%     622078
    15    29.4%     650859

So the default of 6 leaves about a third of the recoverable holes on the table.

NOISE NEEDS A FIXED PIXEL SET TO MEASURE HONESTLY. Comparing the temporal
standard deviation of "pixels valid in every frame" across IR levels is
meaningless, because raising IR changes which pixels those are — it recovers
precisely the marginal, low-texture pixels that are inherently noisiest, so the
aggregate figure gets WORSE while every individual pixel gets better. Measured
on the 429063 pixels valid at both levels:

    IR 6  -> 1.97mm     IR 15 -> 1.35mm      (~31% less noise)

while the 96345 pixels IR 15 recovers, which were holes at IR 6, carry 16.6mm.
Those are noisy — but they are real measurements where there was previously
nothing, and still far better than a fusion estimate. The two settings agree on
depth to a median of 2.2mm, so there is no systematic bias between them.

Expect the newly recovered regions to look grainier than the rest; the SDK
temporal filter earns its keep more at high IR than at low.

Nothing is changed unless `ir_value` is set, because this drives an emitter on
the user's hardware and the sensible level depends on the scene: high IR helps
textureless close-range surfaces but adds nothing beyond the projector's throw,
and reflective surfaces can bloom.

Failures are logged and swallowed — a camera that rejects IR control must still
stream.
"""


def configure(device, cfg):
    """Apply `ir_extended` / `ir_value` from config. No-op when ir_value is unset."""
    value = cfg.get("ir_value")
    if value in (None, "", "null"):
        return

    try:
        ir = device.get_IRProperty()
    except Exception as exc:
        print(f"[ir] IR property unavailable: {exc}", flush=True)
        return

    # Extended mode must be enabled BEFORE setting a value above 6: set_IR_value
    # range-checks against the currently active maximum and raises otherwise.
    if cfg.get("ir_extended", True):
        try:
            ir.enable_extendIR()
        except Exception as exc:
            print(f"[ir] enable_extendIR failed: {exc}", flush=True)

    lo, hi = ir.get_IR_min(), ir.get_IR_max()
    want = int(value)
    clamped = max(lo, min(hi, want))
    if clamped != want:
        print(f"[ir] ir_value {want} outside {lo}..{hi}, using {clamped}", flush=True)

    try:
        ir.set_IR_value(clamped)
        print(f"[ir] IR value set to {ir.get_IR_value()} (range {lo}..{hi}, "
              f"extended={ir.is_extendIR_enabled()})", flush=True)
    except Exception as exc:
        print(f"[ir] set_IR_value({clamped}) failed: {exc}", flush=True)
