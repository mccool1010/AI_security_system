import os

import numpy as np

import action_recognizer as ar

LABELS = ar.load_kinetics_labels(os.path.join(os.path.dirname(ar.__file__), "kinetics_400_labels.json"))


def test_labels_are_clean():
    assert len(LABELS) == 400 and all(LABELS)
    assert not any('"' in l for l in LABELS)


def test_every_category_label_exists():
    index = ar.build_category_index(LABELS)
    assert set(index) == set(ar.CATEGORIES)
    for cat, idx in index.items():
        assert [LABELS[i] for i in idx] == ar.CATEGORIES[cat]["labels"]


def test_noise_classes_are_not_security_relevant():
    members = {l for spec in ar.CATEGORIES.values() for l in spec["labels"]}
    for noisy in ("archery", "hugging", "clapping", "kissing", "trimming or shaving beard", "skateboarding"):
        assert noisy not in members


def test_sample_clip_spans_fixed_time_regardless_of_fps():
    for fps in (5, 12, 30):
        frames = [(i / fps, np.full((2, 2, 3), i, np.uint8)) for i in range(int(4 * fps))]
        clip = ar.ActionRecognizer.sample_clip(frames, frames[-1][0])
        assert clip.shape[0] == ar.CLIP_LEN
        first_ts = clip[0, 0, 0, 0] / fps
        span = frames[-1][0] - first_ts
        assert abs(span - ar.CLIP_SECONDS) < 1.0 / fps + 1e-6


def test_clip_not_rescored_without_new_frames():
    st = ar._Stream()
    for i in range(40):
        st.frames.append((i / 12, np.zeros((2, 2, 3), np.uint8)))
    assert ar.ActionRecognizer.clip_ready(st)
    st.last_frame_ts = st.frames[-1][0]          # what _infer records
    assert not ar.ActionRecognizer.clip_ready(st)
    st.frames.append((41 / 12, np.zeros((2, 2, 3), np.uint8)))
    assert ar.ActionRecognizer.clip_ready(st)


def test_square_roi_is_square_and_inside_frame():
    x1, y1, x2, y2 = ar.square_roi([[600, 400, 630, 470]], (480, 640, 3))
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
    assert abs((x2 - x1) - (y2 - y1)) <= 1
    x1, y1, x2, y2 = ar.square_roi([[10, 10, 50, 100], [500, 50, 560, 200]], (480, 640, 3))
    assert x1 <= 10 and x2 >= 560
