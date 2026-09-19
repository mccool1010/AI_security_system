import numpy as np
import pytest

import detector
import tracking
from pipeline import compute_risk
from tracking import PersonTracker, dwell_label, iou

# Upright skeleton in a unit box: (x, y) per COCO keypoint, y grows downward.
UPRIGHT = {
    0: (0.50, 0.06), 1: (0.47, 0.05), 2: (0.53, 0.05), 3: (0.44, 0.06), 4: (0.56, 0.06),
    5: (0.38, 0.20), 6: (0.62, 0.20), 7: (0.35, 0.35), 8: (0.65, 0.35),
    9: (0.34, 0.48), 10: (0.66, 0.48), 11: (0.43, 0.52), 12: (0.57, 0.52),
    13: (0.43, 0.74), 14: (0.57, 0.74), 15: (0.43, 0.97), 16: (0.57, 0.97),
}


def skeleton(x0, y0, h, rotate=False):
    k = np.zeros((17, 3), np.float32)
    for i, (u, v) in UPRIGHT.items():
        if rotate:  # lying on the ground: swap axes
            u, v = v, u
        k[i] = (x0 + u * h, y0 + v * (h if not rotate else h * 0.35), 0.9)
    return k


def det_for(k):
    xs, ys = k[:, 0], k[:, 1]
    return {"box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            "confidence": 0.9, "keypoints": k.tolist()}


def run(frames):
    tr = PersonTracker()
    out = None
    for ts, k in frames:
        out = tr.update([det_for(k)], ts)
    return tr, out


def test_iou_and_dwell():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert [dwell_label(s) for s in (1, 10, 30, 120)] == ["passing", "visitor", "lingering", "loitering"]


@pytest.mark.parametrize("fps", [5, 10, 30])
def test_standing_vs_walking_independent_of_fps(fps):
    _, still = run([(i / fps, skeleton(100, 50, 300)) for i in range(fps)])
    assert still[0].pose == "standing"
    # 1.0 body-height per second sideways
    _, moving = run([(i / fps, skeleton(100 + 300 * i / fps, 50, 300)) for i in range(fps)])
    assert moving[0].pose == "walking"


@pytest.mark.parametrize("scale", [150, 400])
def test_running_independent_of_person_size(scale):
    fps = 10
    tr, t = run([(i / fps, skeleton(100 + 2.5 * scale * i / fps, 50, scale)) for i in range(fps)])
    assert t[0].pose == "running"
    assert len(tr.tracks) == 1, "fast movement must not split the track"


def test_lying_down():
    _, t = run([(i / 10, skeleton(100, 300, 300, rotate=True)) for i in range(5)])
    assert t[0].pose == "lying_down"


def test_person_cut_off_by_frame_edge_is_not_lying_down():
    # Entering at the bottom edge: only the upper body is visible, box wider than tall,
    # hips barely detected (the false alarm seen on test_video.mp4).
    k = skeleton(430, 330, 60)
    k[[11, 12], 2] = 0.22
    k[[13, 14, 15, 16], 2] = 0.0
    k[:, 1] = np.minimum(k[:, 1], 358)
    tr = PersonTracker()
    t = tr.update([{"box": [435, 330, 479, 359], "confidence": 0.8, "keypoints": k.tolist()}], 0.0,
                  frame_shape=(360, 640, 3))
    assert t[0].truncated
    assert t[0].pose in ("standing", "walking")


def test_urgent_pose_needs_to_persist():
    tr = PersonTracker()
    lying = skeleton(100, 300, 300, rotate=True)
    for i in range(3):  # 0.2 s
        t = tr.update([det_for(lying)], i / 10)
    assert t[0].pose == "lying_down" and t[0].stable_pose != "lying_down"
    for i in range(3, 15):  # > 1 s
        t = tr.update([det_for(lying)], i / 10)
    assert t[0].stable_pose == "lying_down"


def test_falling():
    fps = 10
    frames = [(i / fps, skeleton(100, 50, 300)) for i in range(10)]
    # shoulders drop 0.8 body-heights within 0.4 s while the torso tilts
    for j in range(1, 5):
        k = skeleton(100, 50 + 60 * j, 300)
        k[[5, 6, 0], 0] += 25 * j
        frames.append((1.0 + j / fps, k))
    _, t = run(frames)
    assert t[0].pose == "falling"


def test_two_people_keep_separate_tracks():
    tr = PersonTracker()
    for i in range(10):
        ts = i / 10
        tr.update([det_for(skeleton(50 + 5 * i, 50, 200)), det_for(skeleton(400 - 5 * i, 50, 200))], ts)
    assert len(tr.tracks) == 2


def test_risk_scoring():
    calm = [{"pose": "walking", "dwell": "passing", "identity": None}]
    assert compute_risk(calm, [])[1] == "LOW"
    fall = [{"pose": "falling", "dwell": "passing", "identity": None}]
    assert compute_risk(fall, [])[1] == "HIGH"
    loiter = [{"pose": "standing", "dwell": "loitering", "dwell_s": 75, "identity": {"name": "unknown"}}]
    score, level, reasons = compute_risk(loiter, [])
    assert level == "HIGH" and any("loitering" in r for r in reasons)
    fight = compute_risk(calm, [{"action": "Fighting", "severity": "HIGH", "confidence": 0.6}])
    assert fight[1] == "MEDIUM" and fight[0] == 3
    known = [{"pose": "standing", "dwell": "visitor", "identity": {"name": "Alice"}}]
    assert compute_risk(known, [])[0] == 0


def test_detector_runs_on_blank_frame():
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []


# ── Identity persistence ──────────────────────────────────────────────

def scene(people, size=(360, 640, 3)):
    """Frame with a solid-coloured body per person, so the tracker has appearance to use."""
    frame = np.full(size, 40, np.uint8)
    for (x, y, h, colour) in people:
        w = int(h * 0.4)
        frame[int(y):int(y + h), int(x):int(x + w)] = colour
    return frame


def person_det(x, y, h):
    k = skeleton(x, y, h)
    return det_for(k)


def test_two_people_crossing_keep_their_identities():
    """Paths cross and boxes overlap; motion prediction should carry the IDs through."""
    tr = PersonTracker()
    ids = []
    for i in range(21):
        ts = i / 10
        a_x, b_x = 100 + 10 * i, 300 - 10 * i          # meet in the middle at i=10
        frame = scene([(a_x, 50, 200, (40, 40, 220)), (b_x, 50, 200, (220, 180, 40))])
        tracks = tr.update([person_det(a_x, 50, 200), person_det(b_x, 50, 200)], ts, frame)
        ids.append((tracks[0].id, tracks[1].id))
    assert ids[-1] == ids[0], "identities swapped while the two people crossed"
    assert len(tr.tracks) == 2


def test_track_survives_a_brief_occlusion():
    tr = PersonTracker()
    frame = lambda x: scene([(x, 50, 200, (40, 40, 220))])  # noqa: E731
    for i in range(6):
        t = tr.update([person_det(100 + 20 * i, 50, 200)], i / 10, frame(100 + 20 * i))
    first = t[0].id
    for i in range(6, 12):                                  # hidden for 0.6 s
        tr.update([], i / 10, frame(0))
    t = tr.update([person_det(340, 50, 200)], 1.2, frame(340))
    assert t[0].id == first, "a person hidden briefly should not become a new track"
    assert t[0].dwell_s >= 1.0, "dwell time must survive the occlusion"


def test_velocity_prediction_helps_fast_movers():
    tr = PersonTracker()
    for i in range(4):
        t = tr.update([person_det(50 + 120 * i, 50, 200)], i / 10, scene([(50 + 120 * i, 50, 200, (40, 40, 220))]))
    assert len(tr.tracks) == 1
    assert t[0].velocity[0] > 0
    predicted = t[0].predicted_box(t[0].last_seen + 0.1)
    assert predicted[0] > t[0].box[0], "prediction should move the box along the track's velocity"


def test_appearance_tells_similar_shapes_apart():
    red = tracking.torso_histogram(scene([(100, 50, 200, (40, 40, 220))]), [100, 50, 180, 250])
    red2 = tracking.torso_histogram(scene([(300, 50, 200, (40, 40, 220))]), [300, 50, 380, 250])
    yellow = tracking.torso_histogram(scene([(100, 50, 200, (220, 180, 40))]), [100, 50, 180, 250])
    assert tracking.appearance_similarity(red, red2) > 0.9
    assert tracking.appearance_similarity(red, yellow) < 0.6
    assert tracking.appearance_similarity(red, None) == 0.5      # unknown is neutral


def test_appearance_is_not_learned_while_people_overlap():
    """A crop containing two people must not be blended into either track's model."""
    tr = PersonTracker()
    clean = scene([(100, 50, 200, (40, 40, 220))])
    t = tr.update([person_det(100, 50, 200)], 0.0, clean)[0]
    learned = t.appearance.copy()

    # second person steps onto the same spot: both boxes overlap heavily
    mixed = scene([(100, 50, 200, (40, 40, 220)), (120, 50, 200, (220, 180, 40))])
    tr.update([person_det(100, 50, 200), person_det(120, 50, 200)], 0.1, mixed)
    assert np.allclose(t.appearance, learned), "appearance was polluted by an overlapping person"

    # once clear again, learning resumes
    tr.update([person_det(100, 50, 200)], 0.2, clean)
    assert t.appearance is not None
