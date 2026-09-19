"""Per-camera person tracker.

Association uses three signals, in this order of weight:

  * motion   - each track's box is predicted forward by its velocity before
               matching, so a fast walker at a low frame rate still overlaps
               its own detection;
  * appearance - an HSV histogram of the torso, which keeps identities apart
               when two people cross or one is briefly hidden. It is only
               *learned* from clean views: while two people overlap, each crop
               contains the other, so updating there would blend them together
               and destroy exactly the signal needed to tell them apart;
  * distance - a gate on how far a person can plausibly have moved.

Identity persistence matters beyond tidiness: dwell time (and therefore
loitering) and the cached face identity both live on the track.

Each track also keeps the short keypoint history that posture classification
needs, recent height measurements and its own appearance model.
"""
import os
import uuid
from collections import deque

import cv2
import numpy as np

import detector

STALE_S = 5.0                 # keep unmatched tracks this long, for re-identification
LOITER_S = float(os.environ.get("LOITER_SECONDS", "60"))
DWELL_LABELS = ((5.0, "passing"), (15.0, "visitor"), (LOITER_S, "lingering"))
HISTORY_S = 1.0
EDGE_PX = 3
# how long a posture must persist before it counts for alerts and risk
CONFIRM_S = {"falling": 0.15, "lying_down": 1.0, "fighting": 0.8, "kicking": 0.5}

W_IOU, W_APPEARANCE, W_DISTANCE = 0.60, 0.25, 0.15
MATCH_MIN = 0.25              # below this, start a new track instead
APPEARANCE_ALPHA = 0.3        # EMA on the torso histogram
MIN_APPEARANCE_PX = 24
OVERLAP_AMBIGUOUS = 0.15      # boxes overlapping this much give a mixed appearance crop


def dwell_label(seconds):
    for limit, label in DWELL_LABELS:
        if seconds < limit:
            return label
    return "loitering"


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def torso_histogram(frame, box):
    """HSV histogram of the torso region, or None if the person is too small."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < MIN_APPEARANCE_PX:
        return None
    # middle of the body: skips the head (hair and background) and the legs (often occluded)
    xa, xb = int(max(0, x1 + 0.2 * bw)), int(min(w, x2 - 0.2 * bw))
    ya, yb = int(max(0, y1 + 0.15 * bh)), int(min(h, y1 + 0.6 * bh))
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def appearance_similarity(a, b):
    """Correlation of two histograms mapped to 0..1 (0.5 when either is unknown)."""
    if a is None or b is None:
        return 0.5
    return float(np.clip((cv2.compareHist(a, b, cv2.HISTCMP_CORREL) + 1) / 2, 0, 1))


class Track:
    def __init__(self, box, ts):
        self.id = uuid.uuid4().hex[:8]
        self.first_seen = ts
        self.last_seen = ts
        self.box = box
        self.confidence = 0.0
        self.velocity = np.zeros(2)           # px/s of the box centre
        self.appearance = None
        self.history = deque(maxlen=30)       # (ts, box, kpts ndarray or None)
        self.heights = deque(maxlen=60)       # (ts, height_m)
        self.scale_samples = deque(maxlen=90)  # (ts, longer box side px)
        self.pose = "unknown"
        self.pose_since = ts
        self.stable_pose = "unknown"
        self.truncated = False
        self.identity = None                  # {user_id, name, score}
        self.identity_ts = 0.0
        self.face_attempts = 0
        self.hits = 0

    @property
    def dwell_s(self):
        return self.last_seen - self.first_seen

    @property
    def dwell(self):
        return dwell_label(self.dwell_s)

    def predicted_box(self, ts):
        """Where this track should be now, from its last box and velocity."""
        dt = min(max(0.0, ts - self.last_seen), 1.0)
        if dt == 0 or not self.velocity.any():
            return self.box
        dx, dy = self.velocity * dt
        return [self.box[0] + dx, self.box[1] + dy, self.box[2] + dx, self.box[3] + dy]

    def body_scale(self):
        """Body length in px: high percentile of the recent longer box side (robust to crouching/lying)."""
        if not self.scale_samples:
            return max(self.box[2] - self.box[0], self.box[3] - self.box[1])
        return float(np.percentile([h for _, h in self.scale_samples], 90))

    def update(self, det, ts, frame=None, frame_shape=None, ambiguous=False):
        dt = ts - self.last_seen
        new_box = det["box"]
        if dt > 0:
            v = (centre(new_box) - centre(self.box)) / dt
            self.velocity = v if not self.velocity.any() else 0.5 * v + 0.5 * self.velocity
        self.box = new_box
        self.confidence = float(det.get("confidence", 0.0))
        self.last_seen = ts
        self.hits += 1

        k = det.get("keypoints")
        k = np.asarray(k, dtype=np.float32) if k is not None else None
        self.history.append((ts, self.box, k))
        while self.history and ts - self.history[0][0] > HISTORY_S:
            self.history.popleft()

        shape = frame.shape if frame is not None else frame_shape
        if shape is not None:
            fh, fw = shape[:2]
            b = self.box
            self.truncated = b[0] <= EDGE_PX or b[1] <= EDGE_PX or b[2] >= fw - EDGE_PX or b[3] >= fh - EDGE_PX
        if not self.truncated:
            self.scale_samples.append((ts, max(self.box[2] - self.box[0], self.box[3] - self.box[1])))
            hist = None if ambiguous else torso_histogram(frame, self.box)
            if hist is not None:
                self.appearance = (hist if self.appearance is None
                                   else APPEARANCE_ALPHA * hist + (1 - APPEARANCE_ALPHA) * self.appearance)

        pose = detector.classify_pose(list(self.history), self.body_scale(), self.truncated)
        if pose != self.pose:
            self.pose, self.pose_since = pose, ts
        if ts - self.pose_since >= CONFIRM_S.get(pose, 0.0):
            self.stable_pose = pose

    def height_summary(self):
        """Median of recent measurements and half the interquartile range as spread."""
        if not self.heights:
            return None
        vals = np.array([h for _, h in self.heights])
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        return {"height_m": round(float(med), 3), "spread_m": round(float(q3 - q1) / 2, 3),
                "samples": int(len(vals))}


class PersonTracker:
    def __init__(self):
        self.tracks = {}

    def _score(self, det, track, ts, det_hist):
        """Match quality in 0..1, or None when the detection is out of reach."""
        pred = track.predicted_box(ts)
        overlap = iou(det["box"], pred)
        gap = ts - track.last_seen
        size = max(det["box"][3] - det["box"][1], pred[3] - pred[1], 1.0)
        reach = size * min(3.0, 0.75 + 2.0 * gap)      # how far a person can plausibly move
        dist = float(np.linalg.norm(centre(det["box"]) - centre(pred)))
        if overlap == 0 and dist > reach:
            return None
        looks = appearance_similarity(det_hist, track.appearance)
        return W_IOU * overlap + W_APPEARANCE * looks + W_DISTANCE * max(0.0, 1 - dist / reach)

    def update(self, detections, ts, frame=None, frame_shape=None):
        """Assign detections to tracks. Returns the tracks in detection order."""
        if frame is not None and frame_shape is None:
            frame_shape = frame.shape
        # A detection that overlaps another person gives a mixed crop: usable for
        # comparison, but never for learning.
        ambiguous = [any(i != j and iou(d["box"], o["box"]) > OVERLAP_AMBIGUOUS
                         for j, o in enumerate(detections))
                     for i, d in enumerate(detections)]
        hists = [None if ambiguous[i] else torso_histogram(frame, d["box"])
                 for i, d in enumerate(detections)]

        pairs = []
        for di, d in enumerate(detections):
            for t in self.tracks.values():
                s = self._score(d, t, ts, hists[di])
                if s is not None and s >= MATCH_MIN:
                    pairs.append((s, di, t))
        pairs.sort(key=lambda p: -p[0])

        assigned, used = {}, set()
        for _, di, t in pairs:
            if di in assigned or t.id in used:
                continue
            assigned[di] = t
            used.add(t.id)

        result = []
        for di, d in enumerate(detections):
            t = assigned.get(di)
            if t is None:
                t = Track(d["box"], ts)
                self.tracks[t.id] = t
            t.update(d, ts, frame, frame_shape, ambiguous[di])
            result.append(t)

        for tid in [tid for tid, t in self.tracks.items() if ts - t.last_seen > STALE_S]:
            del self.tracks[tid]
        return result
