import numpy as np
import dataset_utils
from pose.keypoints import PoseData
from config.mouse_config import MouseParadigmConfig, MousePoseSchema
from features import generate_features, feature_registry

def test_legacy_features():
    print("--- Testing Legacy 22-Feature Calculation ---")
    num_frames = 20
    rat_head = np.random.rand(num_frames, 2) * 100
    rat_mid = np.random.rand(num_frames, 2) * 100
    rat_butt = np.random.rand(num_frames, 2) * 100
    rat_tail = np.random.rand(num_frames, 2) * 100
    cricket_pos = np.random.rand(num_frames, 2) * 100
    video_width = 1920
    video_fps = 30.0

    features = dataset_utils.generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)
    print("Legacy feature shape:", features.shape)
    assert features.shape == (num_frames, 22), f"Expected shape ({num_frames}, 22), got {features.shape}"
    assert not np.isnan(features).any(), "NaN found in legacy features"
    print("Legacy feature test PASSED!\n")

def test_mouse_features():
    print("--- Testing Modular Mouse Feature Extraction with Multi-Segment Tail ---")
    num_frames = 50
    video_width = 1024
    video_fps = 60.0  # Testing with 60 FPS mouse video

    # Create dummy PoseData with 4 tail points and ears
    snout = np.random.rand(num_frames, 2) * 500
    center = np.random.rand(num_frames, 2) * 500
    rear = np.random.rand(num_frames, 2) * 500
    tail_pts = [np.random.rand(num_frames, 2) * 500 for _ in range(4)]  # 4 tail segments
    cricket = np.random.rand(num_frames, 2) * 500
    left_ear = np.random.rand(num_frames, 2) * 500
    right_ear = np.random.rand(num_frames, 2) * 500

    pose_data = PoseData(
        num_frames=num_frames,
        head=snout,
        center=center,
        rear=rear,
        tail_points=tail_pts,
        cricket=cricket,
        left_ear=left_ear,
        right_ear=right_ear
    )

    mouse_cfg = MouseParadigmConfig()
    features, manifest = generate_features(
        pose_data=pose_data,
        video_width=video_width,
        video_fps=video_fps,
        enabled_blocks=mouse_cfg.enabled_feature_blocks,
        config=mouse_cfg
    )

    print(f"Mouse features computed: {features.shape[1]} columns across {features.shape[0]} frames.")
    print(f"Feature manifest length: {manifest.num_features}")
    assert features.shape[0] == num_frames
    assert features.shape[1] == manifest.num_features
    assert not np.isnan(features).any(), "NaN found in mouse features"

    print("\nFeature Manifest:")
    for idx, item in enumerate(manifest.items):
        print(f"  Col {idx:02d}: [{item.group:<14}] {item.name:<28} - {item.description}")

    print("\nMouse feature test PASSED!\n")

def test_numerical_equivalence():
    print("--- Testing Bit-for-Bit Numerical Equivalence vs Legacy Rat Pipeline ---")
    np.random.seed(42)
    num_frames = 100
    rat_head = np.random.rand(num_frames, 2) * 200 + 100
    rat_mid = np.random.rand(num_frames, 2) * 200 + 80
    rat_butt = np.random.rand(num_frames, 2) * 200 + 50
    rat_tail = np.random.rand(num_frames, 2) * 200 + 20
    cricket_pos = np.random.rand(num_frames, 2) * 300
    
    # Introduce NaN frames for cricket
    cricket_pos[10:15] = np.nan
    cricket_pos[40:42] = np.nan
    
    video_width = 1920.0
    video_fps = 30.0

    # 1. Baseline legacy formula
    def _dist(p1, p2): return np.linalg.norm(p1 - p2, axis=1)
    def _get_derivs(pos_array):
        vel_vec = np.diff(pos_array, axis=0, prepend=pos_array[0:1]) * video_fps
        speed = np.linalg.norm(vel_vec, axis=1) / video_width
        acc_vec = np.diff(vel_vec, axis=0, prepend=vel_vec[0:1]) * video_fps
        accel = np.linalg.norm(acc_vec, axis=1) / video_width
        return speed, accel

    def _ang_diff(angles):
        d_angles = np.diff(angles, prepend=angles[0:1])
        return (d_angles + np.pi) % (2 * np.pi) - np.pi

    d_head_cricket = _dist(rat_head, cricket_pos) / video_width
    d_head_mid = _dist(rat_head, rat_mid) / video_width
    d_mid_tail = _dist(rat_mid, rat_tail) / video_width
    d_head_tail = _dist(rat_head, rat_tail) / video_width
    
    ba = rat_head - rat_mid
    bc = rat_tail - rat_mid
    cosine_angle = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
    angle_spine = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))) / 180.0
    
    rat_orientation = np.arctan2(rat_head[:, 1] - rat_mid[:, 1], rat_head[:, 0] - rat_mid[:, 0])
    target_vector = np.arctan2(cricket_pos[:, 1] - rat_head[:, 1], cricket_pos[:, 0] - rat_head[:, 0])
    angle_diff_raw = (target_vector - rat_orientation + np.pi) % (2 * np.pi) - np.pi
    angle_diff = angle_diff_raw / np.pi
    
    rat_speed, rat_accel = _get_derivs(rat_head)
    cricket_speed, cricket_accel = _get_derivs(cricket_pos)
    cricket_visible = np.where(np.isnan(d_head_cricket), 0.0, 1.0)
    
    closing_velocity = -np.diff(d_head_cricket, prepend=d_head_cricket[0:1]) * video_fps
    inverse_ttc = np.clip(closing_velocity / (d_head_cricket + 1e-6), -100, 100) 
    rat_turn_rate = np.abs(_ang_diff(rat_orientation)) * video_fps / np.pi
    aim_correction_rate = np.abs(_ang_diff(angle_diff_raw)) * video_fps / np.pi

    vec_hc = cricket_pos - rat_head
    dist_hc = np.linalg.norm(vec_hc, axis=1, keepdims=True)
    unit_hc = vec_hc / (dist_hc + 1e-6)
    rat_vel = np.diff(rat_head, axis=0, prepend=rat_head[0:1]) * video_fps
    lunge_speed = np.sum(rat_vel * unit_hc, axis=1) / video_width

    theta = rat_orientation
    dx = vec_hc[:, 0]
    dy = vec_hc[:, 1]
    ego_forward = (dx * np.cos(-theta) - dy * np.sin(-theta)) / video_width
    ego_lateral = (dx * np.sin(-theta) + dy * np.cos(-theta)) / video_width

    body_compression = d_head_tail / (d_head_mid + d_mid_tail + 1e-6)
    compression_rate = np.diff(body_compression, prepend=body_compression[0:1]) * video_fps

    ba_t = rat_mid - rat_butt
    bc_t = rat_tail - rat_butt
    cos_t = np.sum(ba_t*bc_t, axis=1) / (np.linalg.norm(ba_t, axis=1) * np.linalg.norm(bc_t, axis=1) + 1e-6)
    angle_tail = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))) / 180.0
    tail_angular_vel = np.abs(np.diff(angle_tail, prepend=angle_tail[0:1])) * video_fps

    c_vel = np.diff(cricket_pos, axis=0, prepend=cricket_pos[0:1])
    c_bearing = np.arctan2(c_vel[:, 1], c_vel[:, 0])
    c_jitter = np.abs(_ang_diff(c_bearing)) / np.pi

    MAX_SPEED_THRESH = 3.0
    MAX_ACCEL_THRESH = 25.0
    rat_speed = np.clip(rat_speed, 0, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    cricket_speed = np.clip(cricket_speed, 0, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    rat_accel = np.clip(rat_accel, 0, MAX_ACCEL_THRESH) / MAX_ACCEL_THRESH
    cricket_accel = np.clip(cricket_accel, 0, MAX_ACCEL_THRESH) / MAX_ACCEL_THRESH
    closing_velocity = np.clip(closing_velocity, -MAX_SPEED_THRESH, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    lunge_speed = np.clip(lunge_speed, -MAX_SPEED_THRESH, MAX_SPEED_THRESH) / MAX_SPEED_THRESH

    inverse_ttc = np.tanh(inverse_ttc / 10.0)
    compression_rate = np.tanh(compression_rate / 10.0)
    tail_angular_vel = np.tanh(tail_angular_vel / 5.0)
    rat_turn_rate = np.tanh(rat_turn_rate / 10.0)
    aim_correction_rate = np.tanh(aim_correction_rate / 10.0)

    expected = np.column_stack([
        d_head_cricket, d_head_mid, d_mid_tail, d_head_tail,
        angle_spine, angle_diff, rat_speed, rat_accel,
        cricket_speed, cricket_accel, cricket_visible,
        closing_velocity, inverse_ttc, rat_turn_rate, aim_correction_rate,
        lunge_speed, ego_forward, ego_lateral, body_compression,
        compression_rate, tail_angular_vel, c_jitter
    ])
    expected = np.nan_to_num(expected, nan=0.0)

    # 2. Compute with new modular dataset_utils.generate_features
    actual = dataset_utils.generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)

    diff = np.max(np.abs(expected - actual))
    print(f"Max absolute difference vs baseline: {diff:.2e}")
    assert np.allclose(expected, actual, atol=1e-12), f"Equivalence test failed! Max diff: {diff}"
    print("Exact numerical equivalence test PASSED (difference < 1e-12)!\n")

def test_diane_label_parsing_and_strategies():
    print("--- Testing Diane Format Master Label Parsing & Strategies ---")
    import pandas as pd
    from labels.diane_parser import DianeExcelLabelParser

    # Construct test dataframe mimicking Diane's master sheet from the user's screenshot
    data = {
        'Animal': [158, np.nan, np.nan, 'YB-12', np.nan],
        'Day': [1, 1, 1, 2, 2],
        'Cricket': [1, 2, 3, 1, 2],
        'Released': [6450, 10783, 15997, 5000, 9000],
        'First approach': [7170, 11386, 16423, 5200, 9200],
        'Final approach': [7170, 13614, 16619, 5200, 11500],
        'Capture': [7237, 13650, 16785, 5300, 11550],
        'Eaten': [8073, 14061, 17290, 6000, 12200],
    }
    df = pd.DataFrame(data)
    test_csv = "test_diane_sheet.csv"
    df.to_csv(test_csv, index=False)

    try:
        parser = DianeExcelLabelParser(tentative_approach_max_sec=2.0, default_fps=30.0, tentative_label_state=1)
        hunts_map = parser.parse_file(test_csv)

        # Verify animal normalization and forward-fill
        assert ('158', 1) in hunts_map, f"Expected ('158', 1) in hunts_map keys: {list(hunts_map.keys())}"
        assert ('YB-12', 2) in hunts_map, f"Expected ('YB-12', 2) in hunts_map keys: {list(hunts_map.keys())}"

        session_158 = hunts_map[('158', 1)]
        assert len(session_158) == 3, f"Expected 3 hunts for 158 Day 1, got {len(session_158)}"

        # Test Strategy 1: 'engagement' (Default continuous multi-approach engagement)
        events_eng = parser.hunts_to_events(session_158, tentative_approach_strategy="engagement")
        assert len(events_eng) == 3, f"Expected 3 events under 'engagement' strategy, got {len(events_eng)}"
        # Cricket 2: Stalking from 11386 to 13614, Pursuit from 13614 to 13650, Eating from 13650 to 14061
        assert events_eng[1].enter == 11386
        assert events_eng[1].approach_start == 13614
        assert events_eng[1].capture == 13650
        assert events_eng[1].consumption_end == 14061

        # Test Strategy 2: 'split' (experimental fixed-cap split)
        events_split = parser.hunts_to_events(session_158, tentative_approach_strategy="split")
        assert len(events_split) == 5, f"Expected 5 events under 'split' strategy, got {len(events_split)}"

        # Check Cricket 2 split details
        # First approach: 11386, Final approach: 13614, max tentative frames: 60 (2s * 30fps)
        ev_aborted = events_split[1]
        assert ev_aborted.metadata.get('type') == 'aborted_stalk'
        assert ev_aborted.enter == 11386
        assert ev_aborted.approach_start == 11386 + 60
        assert ev_aborted.capture == 11386 + 60

        ev_final = events_split[2]
        assert ev_final.metadata.get('type') == 'final_capture'
        assert ev_final.enter == 13614
        assert ev_final.approach_start == 13614
        assert ev_final.capture == 13650
        assert ev_final.consumption_end == 14061

        # Test Strategy 2: 'continuous' (solid block)
        events_cont = parser.hunts_to_events(session_158, tentative_approach_strategy="continuous")
        assert len(events_cont) == 3, f"Expected 3 events under 'continuous' strategy, got {len(events_cont)}"
        assert events_cont[1].enter == 11386
        assert events_cont[1].approach_start == 13614

        # Test Strategy 3: 'final_only'
        events_final = parser.hunts_to_events(session_158, tentative_approach_strategy="final_only")
        assert len(events_final) == 3
        assert events_final[1].enter == 13614
        assert events_final[1].approach_start == 13614

        print("Diane format parsing and strategy conversion PASSED!\n")
    finally:
        import os
        if os.path.exists(test_csv):
            os.remove(test_csv)

def test_distal_tail_and_mouse_features():
    print("--- Testing Distal 3-Point Tail Angle & Mouse Dynamics ---")
    num_frames = 60
    video_width = 800.0
    video_fps = 30.0

    snout = np.zeros((num_frames, 2))
    snout[:, 0] = np.linspace(100, 400, num_frames)  # Moving steadily to the right
    snout[:, 1] = 200.0

    center = snout - np.array([40.0, 0.0])
    rear = snout - np.array([80.0, 0.0])

    # 5 tail points: tail1, tail2, tail3, near_endtail, endtail
    # Let's create a distinct bend at the distal tail: tail3, near_endtail, endtail
    tail1 = rear - np.array([20.0, 0.0])
    tail2 = rear - np.array([40.0, 0.0])
    tail3 = rear - np.array([60.0, 0.0])
    near_endtail = tail3 - np.array([15.0, 10.0])   # Bent upward/right
    endtail = near_endtail - np.array([15.0, -10.0]) # Bent down again

    cricket = np.zeros((num_frames, 2))
    cricket[:, 0] = 500.0  # Directly ahead of snout
    cricket[:, 1] = 200.0

    left_ear = snout - np.array([10.0, 15.0])
    right_ear = snout - np.array([10.0, -15.0])

    pose = PoseData(
        num_frames=num_frames,
        head=snout,
        center=center,
        rear=rear,
        tail_points=[tail1, tail2, tail3, near_endtail, endtail],
        cricket=cricket,
        left_ear=left_ear,
        right_ear=right_ear
    )

    mouse_cfg = MouseParadigmConfig()
    features, manifest = generate_features(
        pose_data=pose,
        video_width=video_width,
        video_fps=video_fps,
        enabled_blocks=mouse_cfg.enabled_feature_blocks,
        config=mouse_cfg
    )

    # Check that distal 3pt tail angle is present
    feat_names = [item.name for item in manifest.items]
    assert "distal_3pt_tail_angle" in feat_names, "Missing 'distal_3pt_tail_angle' in feature manifest!"
    assert "distal_3pt_tail_vel" in feat_names, "Missing 'distal_3pt_tail_vel' in feature manifest!"
    assert "approach_directness" in feat_names, "Missing 'approach_directness' in feature manifest!"
    assert "meander_index" in feat_names, "Missing 'meander_index' in feature manifest!"
    assert "sensitive_closing_rate" in feat_names, "Missing 'sensitive_closing_rate' in feature manifest!"
    assert "body_compactness" in feat_names, "Missing 'body_compactness' in feature manifest!"

    idx_distal = feat_names.index("distal_3pt_tail_angle")
    idx_direct = feat_names.index("approach_directness")
    idx_meander = feat_names.index("meander_index")

    # Since snout is moving directly at the cricket, approach directness should be close to 1.0!
    mean_directness = np.mean(features[1:, idx_direct])
    print(f"Mean approach directness (moving straight at cricket): {mean_directness:.3f}")
    assert mean_directness > 0.9, f"Expected approach directness > 0.9, got {mean_directness}"

    # Path is a straight line, so meander index should be very low (~0)
    mean_meander = np.mean(features[10:, idx_meander])
    print(f"Mean meander index (straight path): {mean_meander:.3f}")
    assert mean_meander < 0.1, f"Expected meander index < 0.1, got {mean_meander}"

    # Distal angle should be in valid normalized range [0, 1]
    distal_angles = features[:, idx_distal]
    assert np.all(distal_angles >= 0.0) and np.all(distal_angles <= 1.0)
    print(f"Distal 3-point angle computed successfully: {distal_angles[0]:.3f}")

    print("Distal 3-point tail angle & mouse dynamics test PASSED!\n")

def test_temporal_fps_resampling():
    print("--- Testing Canonical FPS Resampling (20 FPS -> 30 FPS) ---")
    orig_fps = 20.0
    target_fps = 30.0
    num_orig_frames = 100  # 5.0 seconds of data

    # Create dummy linear feature stream: f(t) = t
    features_20fps = np.linspace(0, 5.0, num_orig_frames)[:, np.newaxis]

    resampled_30fps = dataset_utils.resample_feature_matrix(features_20fps, orig_fps=orig_fps, target_fps=target_fps)
    expected_target_frames = round(num_orig_frames * (target_fps / orig_fps))  # 150 frames

    assert len(resampled_30fps) == expected_target_frames, f"Expected {expected_target_frames} frames, got {len(resampled_30fps)}"
    assert np.isclose(resampled_30fps[0, 0], 0.0)
    assert np.isclose(resampled_30fps[-1, 0], 5.0)

    # Test hunt event scaling
    hunts_20fps = [
        {'enter': 20, 'pursuit_start': 40, 'capture': 60, 'consumption_end': 80}
    ]
    scaled_hunts = dataset_utils.scale_hunt_events_fps(hunts_20fps, orig_fps=orig_fps, target_fps=target_fps)
    assert scaled_hunts[0]['enter'] == 30      # 20 * 1.5 = 30
    assert scaled_hunts[0]['pursuit_start'] == 60  # 40 * 1.5 = 60
    assert scaled_hunts[0]['capture'] == 90    # 60 * 1.5 = 90
    assert scaled_hunts[0]['consumption_end'] == 120 # 80 * 1.5 = 120

    print("Canonical FPS temporal resampling test PASSED!\n")

if __name__ == "__main__":
    test_legacy_features()
    test_numerical_equivalence()
    test_mouse_features()
    test_distal_tail_and_mouse_features()
    test_diane_label_parsing_and_strategies()
    test_temporal_fps_resampling()
