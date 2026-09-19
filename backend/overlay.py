"""Drawing the pipeline's results onto the video stream.

Text is ASCII only: OpenCV's built-in Hershey fonts draw anything else as "?".

The worker publishes a small, already-computed description of what it found
(`build_state`); the reader thread draws it onto the newest frame just before
JPEG encoding, and only while somebody is watching. Drawing never touches the
frame the worker analyses — it always works on a copy, so overlay pixels can
never feed back into detection.

Boxes are coloured by what the person contributes to the risk score, so the
stream shows the same judgement as the events list.
"""
import cv2
import numpy as np

# COCO pairs: arms, torso, legs, face
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
KP_CONF_MIN = 0.3

WHITE = (255, 255, 255)
DIM = (170, 170, 170)
COLOURS = {                      # BGR
    "alert": (60, 60, 235),      # falling, lying down, fighting
    "watch": (40, 170, 245),     # loitering, kicking
    "known": (120, 200, 90),     # recognised person
    "normal": (225, 170, 70),
}
URGENT = {"falling", "lying_down", "fighting"}
WATCH = {"kicking", "crouching"}


def person_colour(p):
    pose = p.get("stable_pose") or p.get("pose")
    if pose in URGENT:
        return COLOURS["alert"]
    if pose in WATCH or p.get("dwell") == "loitering":
        return COLOURS["watch"]
    identity = p.get("identity")
    if identity and identity.get("name") not in (None, "unknown"):
        return COLOURS["known"]
    return COLOURS["normal"]


def person_labels(p):
    """Two short lines: who, and what they are doing."""
    identity = p.get("identity")
    who = (identity or {}).get("name")
    who = who if who and who != "unknown" else f"#{p.get('track_id', '')[:4]}"
    first = who
    if p.get("height_m"):
        first += f"  {p['height_m']:.2f}m"
        if p.get("height_spread_m") is not None:
            first += f" +/-{round(p['height_spread_m'] * 100)}cm"
    pose = (p.get("stable_pose") or p.get("pose") or "").replace("_", " ")
    second = " - ".join(x for x in (pose, p.get("dwell"), f"{p['dwell_s']:.0f}s" if p.get("dwell_s") else "") if x)
    return first, second


def build_state(tracks, persons, snapshot, name, trail_s=2.0):
    """Compact, picklable description of what to draw (called by the worker)."""
    people = []
    for tr, p in zip(tracks, persons):
        k = tr.history[-1][2] if tr.history else None
        end = tr.history[-1][0] if tr.history else 0
        trail = [((b[0] + b[2]) / 2, b[3]) for t, b, _ in tr.history if end - t <= trail_s]
        people.append({
            "box": tr.box,
            "keypoints": None if k is None else np.asarray(k, dtype=np.float32),
            "trail": trail,
            "colour": person_colour(p),
            "labels": person_labels(p),
        })
    return {
        "people": people,
        "camera": name,
        "risk": snapshot.get("risk_level", "LOW"),
        "reasons": snapshot.get("risk_reasons", [])[:3],
        "activities": [f"{a['action']} {a['confidence']:.0%}" for a in snapshot.get("activities", [])[:2]],
        "ts": snapshot.get("ts"),
    }


def _label(frame, text, x, y, colour, scale=0.42, thickness=1):
    """Text on a filled chip, clipped into the frame. Non-ASCII is stripped."""
    text = text.encode("ascii", "ignore").decode()
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    h, w = frame.shape[:2]
    x = int(min(max(0, x), max(0, w - tw - 8)))
    y = int(min(max(th + 6, y), h - 4))
    cv2.rectangle(frame, (x, y - th - base - 3), (x + tw + 8, y + 2), colour, -1)
    cv2.putText(frame, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw(frame, state, skeletons=True):
    """Draw `state` onto `frame` (modifies it). Pass a copy, never the analysed frame."""
    if not state:
        return frame
    h, w = frame.shape[:2]
    for p in state["people"]:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        colour = p["colour"]

        if len(p["trail"]) > 2:                       # where they have walked
            pts = np.array(p["trail"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, colour, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        k = p["keypoints"]
        if skeletons and k is not None:
            for a, b in SKELETON:
                if k[a][2] > KP_CONF_MIN and k[b][2] > KP_CONF_MIN:
                    cv2.line(frame, (int(k[a][0]), int(k[a][1])), (int(k[b][0]), int(k[b][1])),
                             colour, 1, cv2.LINE_AA)
            for x, y, c in k:
                if c > KP_CONF_MIN:
                    cv2.circle(frame, (int(x), int(y)), 2, WHITE, -1, cv2.LINE_AA)

        first, second = p["labels"]
        _label(frame, first, x1, y1 - 4, colour)
        if second:
            _label(frame, second, x1, min(y2 + 16, h - 2), colour, scale=0.38)

    # header: camera, people, risk and why
    header = f"{state['camera']}   {len(state['people'])} tracked"
    if state["activities"]:
        header += "   " + " / ".join(state["activities"])
    cv2.rectangle(frame, (0, 0), (w, 22), (28, 28, 32), -1)
    cv2.putText(frame, header.encode("ascii", "ignore").decode(), (8, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)

    risk = state["risk"]
    risk_colour = {"HIGH": COLOURS["alert"], "MEDIUM": COLOURS["watch"]}.get(risk, COLOURS["known"])
    _label(frame, f"RISK {risk}", w - 90, 17, risk_colour, scale=0.45)
    if risk != "LOW" and state["reasons"]:
        _label(frame, " / ".join(state["reasons"]), 0, h - 6, risk_colour, scale=0.4)
    return frame
