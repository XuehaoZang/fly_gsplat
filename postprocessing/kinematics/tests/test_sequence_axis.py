"""Verification for `correct_body_axis/sequence_axis.py` (chain sign
determination + mandatory anchor safety net) and `body_frame.py`'s new
`x_body` override.

`build_guide_chain`/`verify_and_flip_by_anchors` are pure functions over
`{frame_id: axis}`/`{frame_id: body_xyz}` dicts, so most of this is testable
without any real dataset -- only `test_build_guide_chain_smoke` touches
`mock.py` synthetic point clouds, and nothing here touches the real 640-frame
dataset (see the conversation record / `correct_body_axis/diag/` for that
validation).

Runnable both under pytest and standalone: `python test_sequence_axis.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from postprocessing.kinematics import body_frame as bf
from postprocessing.kinematics import io_schema, mock
from postprocessing.kinematics.correct_body_axis.sequence_axis import (
    build_guide_chain,
    verify_and_flip_by_anchors,
)


def _angular_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# body_frame.py: x_body override
# ---------------------------------------------------------------------------


def test_x_body_override_fixes_negative_pitch_head_sign_failure():
    """`test_s2.py::test_negative_pitch_head_sign_heuristic_documented_failure`
    shows the no-`x_body` heuristic flips sign for a nose-down body (picks
    tail as head). Supplying the mock's own true `x_body` (what a correct
    sequence-level estimate should converge to) must bypass that heuristic
    entirely and recover the true yaw/pitch."""
    case = dict(yaw_deg=30.0, pitch_deg=-15.0, roll_deg=10.0)
    gt = mock.default_ground_truth(**case)
    df, _ = mock.make_frame(gt, seed=0)
    true_x_body, _true_y, _true_z = mock.body_axes(gt)

    frame = bf.estimate_body_frame(
        io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df),
        x_body=true_x_body,
    )
    assert abs(_angular_diff_deg(frame.yaw, case["yaw_deg"])) < 3.0, frame.yaw
    assert abs(_angular_diff_deg(frame.pitch, case["pitch_deg"])) < 3.0, frame.pitch


def test_x_body_override_is_normalized_and_used_verbatim():
    df, gt = mock.scenario_clean(seed=0)
    raw = np.array([2.0, 0.0, 0.0])  # not unit, points along +x
    frame = bf.estimate_body_frame(
        io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df),
        x_body=raw,
    )
    assert abs(np.linalg.norm(frame.x_body) - 1.0) < 1e-12
    assert np.allclose(frame.x_body, [1.0, 0.0, 0.0])
    assert abs(frame.yaw - 0.0) < 1e-9
    assert abs(frame.pitch - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# verify_and_flip_by_anchors: pure logic, synthetic axes
# ---------------------------------------------------------------------------


def _axis(deg_from_x: float) -> np.ndarray:
    """Unit vector in the xy-plane, `deg_from_x` degrees from +x."""
    rad = np.radians(deg_from_x)
    return np.array([np.cos(rad), np.sin(rad), 0.0])


def test_verify_and_flip_backfills_first_segment_when_anchor_disagrees():
    """A 4-frame chain, one monolithic (wrong) branch throughout (mirrors the
    real-dataset finding: 0 internal chain flips, but the whole thing is 180
    deg off from the trusted anchor). Frame 0 is itself a reset checkpoint
    (not an anchor) and is trusted as-is -- it owns only its own segment, no
    retroactive check against a *later* anchor's disagreement (this is the
    one documented seam the real-dataset validation also found, at exactly
    this frame-0/frame-1 boundary). The anchor at frame 2 then backfills
    frames 1-2 (everything since the previous checkpoint), and frame 3 (past
    the only anchor) inherits the same correction via tail extension."""
    frame_ids = [0, 1, 2, 3]
    wrong_branch = {fid: _axis(5.0 * fid) for fid in frame_ids}  # smooth, near +x
    chain_diag = {
        0: {"method": "anchor_guide"},
        1: {"method": "prev_frame_guide"},
        2: {"method": "prev_frame_guide"},
        3: {"method": "prev_frame_guide"},
    }
    anchor_axes = {2: _axis(10.0 + 180.0)}  # truth is near -x at frame 2

    corrected, events = verify_and_flip_by_anchors(frame_ids, wrong_branch, chain_diag, anchor_axes)

    assert np.allclose(corrected[0], wrong_branch[0])  # reset checkpoint: trusted as-is
    for fid in (1, 2, 3):
        assert np.allclose(corrected[fid], -wrong_branch[fid]), fid
    assert len(events) == 2  # segment [1,2] flip + tail-extension flip for frame 3
    assert events[0]["segment_start_frame_id"] == 1
    assert events[0]["segment_end_frame_id"] == 2
    assert events[0]["n_frames_flipped"] == 2
    assert events[1]["segment_start_frame_id"] == 3
    assert events[1]["n_frames_flipped"] == 1


def test_verify_and_flip_no_change_when_chain_already_agrees_with_anchor():
    frame_ids = [0, 1, 2]
    axes = {fid: _axis(5.0 * fid) for fid in frame_ids}
    chain_diag = {fid: {"method": "prev_frame_guide"} for fid in frame_ids}
    chain_diag[0] = {"method": "anchor_guide"}
    anchor_axes = {2: _axis(10.0)}  # same branch as the chain

    corrected, events = verify_and_flip_by_anchors(frame_ids, axes, chain_diag, anchor_axes)

    for fid in frame_ids:
        assert np.allclose(corrected[fid], axes[fid]), fid
    assert events == []


def test_verify_and_flip_segments_are_independent():
    """Two anchors: the chain disagrees with the first, agrees with the
    second (evaluated against each segment's own *original*, uncorrected
    chain value) -- only the first segment should flip."""
    frame_ids = [0, 1, 2, 3, 4]
    axes = {fid: _axis(5.0 * fid) for fid in frame_ids}  # one smooth branch
    chain_diag = {fid: {"method": "prev_frame_guide"} for fid in frame_ids}
    chain_diag[0] = {"method": "anchor_guide"}
    anchor_axes = {
        1: _axis(5.0 + 180.0),   # disagrees -> segment [0,1] flips
        4: _axis(20.0),          # agrees with original axes[4] -> no flip
    }

    corrected, events = verify_and_flip_by_anchors(frame_ids, axes, chain_diag, anchor_axes)

    assert np.allclose(corrected[0], axes[0])  # reset checkpoint: trusted as-is
    assert np.allclose(corrected[1], -axes[1])
    for fid in (2, 3, 4):
        assert np.allclose(corrected[fid], axes[fid]), fid
    assert len(events) == 1
    assert events[0]["segment_end_frame_id"] == 1


def test_verify_and_flip_skips_nan_frames():
    frame_ids = [0, 1, 2]
    axes = {0: _axis(0.0), 1: np.full(3, np.nan), 2: _axis(10.0)}
    chain_diag = {
        0: {"method": "anchor_guide"},
        1: {"method": "failed:degenerate"},
        2: {"method": "prev_frame_guide"},
    }
    anchor_axes = {2: _axis(10.0 + 180.0)}

    corrected, events = verify_and_flip_by_anchors(frame_ids, axes, chain_diag, anchor_axes)

    assert np.all(np.isnan(corrected[1]))
    assert np.allclose(corrected[0], axes[0])  # reset checkpoint: trusted as-is
    assert np.allclose(corrected[2], -axes[2])


# ---------------------------------------------------------------------------
# build_guide_chain: smoke test against mock point clouds
# ---------------------------------------------------------------------------


def test_build_guide_chain_smoke_is_continuous_and_finite():
    """Three frames, body yawing smoothly by 10 deg each step -- no gaps, no
    failures expected. Chain should stay finite/unit and, per
    `compute_robust_x_body`'s own geometric guarantee, never flip against
    its own immediately-preceding guide."""
    frame_ids = [0, 1, 2]
    body_xyz_by_frame = {}
    for i, fid in enumerate(frame_ids):
        gt = mock.default_ground_truth(yaw_deg=10.0 * i, pitch_deg=5.0, roll_deg=0.0)
        df, _ = mock.make_frame(gt, seed=fid)
        body_xyz_by_frame[fid] = io_schema.body_xyz(df)

    axes, diags = build_guide_chain(frame_ids, body_xyz_by_frame, anchor_axes={})

    assert set(axes) == set(frame_ids)
    for fid in frame_ids:
        assert np.all(np.isfinite(axes[fid]))
        assert abs(np.linalg.norm(axes[fid]) - 1.0) < 1e-9

    assert diags[0]["method"] == "pca_up_fallback"  # no prev, no anchor -> last-resort fallback
    assert diags[1]["method"] == "prev_frame_guide"
    assert diags[2]["method"] == "prev_frame_guide"

    dot01 = float(np.dot(axes[0], axes[1]))
    dot12 = float(np.dot(axes[1], axes[2]))
    assert dot01 >= 0.0, dot01
    assert dot12 >= 0.0, dot12


def test_build_guide_chain_resets_after_frame_id_gap():
    frame_ids = [0, 5]  # gap > 1
    body_xyz_by_frame = {}
    for fid in frame_ids:
        gt = mock.default_ground_truth(yaw_deg=0.0, pitch_deg=5.0, roll_deg=0.0)
        df, _ = mock.make_frame(gt, seed=fid)
        body_xyz_by_frame[fid] = io_schema.body_xyz(df)

    axes, diags = build_guide_chain(frame_ids, body_xyz_by_frame, anchor_axes={})
    assert diags[0]["method"] == "pca_up_fallback"
    assert diags[5]["method"] == "pca_up_fallback"  # reset, not prev_frame_guide


def _run_all():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
