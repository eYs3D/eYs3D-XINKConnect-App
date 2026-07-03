# Vendored BYTETracker (derived from Ultralytics YOLO, AGPL-3.0 license).
#
# Only the subset BYTETracker needs is included so the component does not have
# to pip-install the full `ultralytics` package (whose `lap` dependency has no
# wheel for py3.8/aarch64 and fails to build on the device). The algorithm is
# unchanged from ultralytics 8.0.238; the linear assignment uses scipy instead
# of lap (ultralytics' own fallback path).

from .byte_tracker import BYTETracker

__all__ = ["BYTETracker"]
