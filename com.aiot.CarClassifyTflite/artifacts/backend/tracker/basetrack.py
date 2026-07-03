# Vendored from Ultralytics YOLO 🚀, AGPL-3.0 license (ultralytics 8.0.238).
# This module defines the base classes and structures for object tracking.

from collections import OrderedDict

import numpy as np


class TrackState:
    """
    Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.
        Tracked (int): State when the object is successfully tracked in subsequent frames.
        Lost (int): State when the object is no longer tracked.
        Removed (int): State when the object is removed from tracking.
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for object tracking, providing foundational attributes and methods."""

    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """Return the last frame ID of the track."""
        return self.frame_id

    @staticmethod
    def next_id():
        """Increment and return the global track ID counter."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """Abstract method to activate the track with provided arguments."""
        raise NotImplementedError

    def predict(self):
        """Abstract method to predict the next state of the track."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Abstract method to update the track with new observations."""
        raise NotImplementedError

    def mark_lost(self):
        """Mark the track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark the track as removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """Reset the global track ID counter."""
        BaseTrack._count = 0
