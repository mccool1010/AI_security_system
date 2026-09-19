import math
import random

import pytest

import height_geometry as hg

FY, CY = 600.0, 240.0


def project(H, theta, D, h):
    """Image rows of a person of height h standing at ground distance D."""
    def row(z):
        phi = math.atan2(H - z, D)          # angle below horizontal to a point at height z
        return CY + FY * math.tan(phi - theta)
    return row(0.0), row(h)


def test_height_from_rows_inverts_projection():
    H, th = 3.0, math.radians(25)
    for D, h in [(3.0, 1.55), (6.0, 1.82), (10.0, 1.70), (4.5, 1.20)]:
        vf, vh = project(H, th, D, h)
        est_h, est_D = hg.height_from_rows(vf, vh, H, th, FY, CY)
        assert est_h == pytest.approx(h, abs=1e-6)
        assert est_D == pytest.approx(D, abs=1e-6)


def test_feet_above_horizon_is_rejected():
    assert hg.height_from_rows(CY - 300, CY - 400, 3.0, math.radians(5), FY, CY) == (None, None)


def test_fit_recovers_camera_and_measures_other_people():
    H, th = 2.8, math.radians(20)
    rng = random.Random(0)
    samples = []
    for D in (2.5, 3.5, 5.0, 7.0, 9.0):
        vf, vh = project(H, th, D, 1.78)
        samples.append({"v_foot": vf + rng.gauss(0, 0.5), "v_head": vh + rng.gauss(0, 0.5), "height_m": 1.78})
    res = hg.fit(samples, FY, CY)
    assert res["camera_height_m"] == pytest.approx(H, abs=0.05)
    assert res["tilt_deg"] == pytest.approx(20, abs=0.5)
    assert res["loo_rms_cm"] is not None and res["loo_rms_cm"] < 3

    # A different person is measured as themselves, not as the reference height.
    vf, vh = project(H, th, 4.0, 1.52)
    h, _ = hg.height_from_rows(vf, vh, res["camera_height_m"], math.radians(res["tilt_deg"]), FY, CY)
    assert h == pytest.approx(1.52, abs=0.03)


def test_fit_with_known_camera_height_needs_one_sample():
    H, th = 3.2, math.radians(30)
    vf, vh = project(H, th, 5.0, 1.70)
    res = hg.fit([{"v_foot": vf, "v_head": vh, "height_m": 1.70}], FY, CY, camera_height_m=H)
    assert res["tilt_deg"] == pytest.approx(30, abs=0.01)


def test_fit_requires_enough_samples():
    with pytest.raises(ValueError):
        hg.fit([{"v_foot": 400, "v_head": 100, "height_m": 1.7}], FY, CY)


def test_ground_calibration_rescales_to_current_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(hg, "CALIB_DIR", str(tmp_path))
    H, th = 3.0, math.radians(22)
    g = hg.GroundCalibration("camX")
    for D in (3, 5, 8):
        vf, vh = project(H, th, D, 1.75)
        # recorded at 640x480
        g.add_sample({"v_foot": vf, "v_head": vh, "height_m": 1.75, "resolution": [640, 480]})
    g.solve(FY, CY, (640, 480))
    vf, vh = project(H, th, 6, 1.60)
    # same scene at 1280x960: every row doubles
    h, _ = g.measure(vf * 2, vh * 2, (960, 1280, 3))
    assert h == pytest.approx(1.60, abs=0.02)
    assert hg.GroundCalibration("camX").is_calibrated  # persisted
