"""Inference thread: pull frames from queue → detect → annotate → write last_frame."""
import queue
import threading


class InferenceThread(threading.Thread):
    def __init__(self, in_queue, last_frame_holder, detector,
                 annotate_fn, det_every_n_frames=1):
        super().__init__(daemon=True, name="InferenceThread")
        self._q = in_queue
        self._holder = last_frame_holder
        self._detector = detector
        self._annotate = annotate_fn
        self._every_n = max(1, int(det_every_n_frames))
        self._running = True
        self._frame_no = 0
        self._last_result = None  # (boxes, scores, kps)

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._frame_no += 1

            if self._detector is None:
                self._holder.set(frame)
                continue

            if self._frame_no % self._every_n == 0:
                try:
                    self._last_result = self._detector.detect(frame)
                except Exception:
                    self._last_result = None

            if self._last_result is not None:
                boxes, scores, kps = self._last_result
                if len(boxes):
                    self._annotate(frame, boxes, scores, kps)
            self._holder.set(frame)
