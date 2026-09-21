import os
import tempfile
import numpy as np
import pandas as pd
import torch

from pose.keypoints import PoseData
from pose.dlc_adapter import DLCAdapter
from config.mouse_config import MouseParadigmConfig
from features import generate_features
from MouseHunt_Inference import MouseInferencePipeline

def test_legacy_boolean_mode():
    """Validates that continuous_cricket_confidence=False reproduces legacy binary cutoff behavior."""
    num_frames = 10
    head = np.ones((num_frames, 2)) * 100.0
    center = np.ones((num_frames, 2)) * 80.0
    rear = np.ones((num_frames, 2)) * 60.0
    tail_points = [np.ones((num_frames, 2)) * (50.0 - i * 10) for i in range(5)]
    
    # Cricket positions: frames 0..6 valid coords, 7..9 NaN
    cricket = np.ones((num_frames, 2)) * 200.0
    cricket[7:] = np.nan
    
    conf_values = np.array([0.95, 0.75, 0.50, 0.35, 0.15, 0.80, 1.0, 0.90, 0.50, 0.0], dtype=np.float32)
    
    pose = PoseData(
        num_frames=num_frames,
        head=head,
        center=center,
        rear=rear,
        tail_points=tail_points,
        cricket=cricket,
        confidences={'cricket': conf_values}
    )
    
    # Default config has continuous_cricket_confidence = False
    config = MouseParadigmConfig()
    assert config.continuous_cricket_confidence is False, "Default must be False for backwards compatibility"
    
    matrix, manifest = generate_features(
        pose_data=pose,
        video_width=514.0,
        video_fps=30.0,
        enabled_blocks=config.enabled_feature_blocks,
        config=config
    )
    
    vis_idx = manifest.feature_names.index("cricket_visible")
    vis_feature = matrix[:, vis_idx]
    
    # Frames 0..6 MUST be exactly 1.0, and 7..9 MUST be 0.0
    expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(vis_feature, expected, atol=1e-5)
    print("Test 1: Legacy boolean mode feature test PASSED!")

def test_continuous_mode_feature():
    """Validates that continuous_cricket_confidence=True outputs continuous confidence values."""
    num_frames = 10
    head = np.ones((num_frames, 2)) * 100.0
    center = np.ones((num_frames, 2)) * 80.0
    rear = np.ones((num_frames, 2)) * 60.0
    tail_points = [np.ones((num_frames, 2)) * (50.0 - i * 10) for i in range(5)]
    
    # Cricket positions: frames 0..6 valid coords, 7..9 NaN
    cricket = np.ones((num_frames, 2)) * 200.0
    cricket[7:] = np.nan
    
    conf_values = np.array([0.95, 0.75, 0.50, 0.35, 0.15, 0.80, 1.0, 0.90, 0.50, 0.0], dtype=np.float32)
    
    pose = PoseData(
        num_frames=num_frames,
        head=head,
        center=center,
        rear=rear,
        tail_points=tail_points,
        cricket=cricket,
        confidences={'cricket': conf_values}
    )
    
    config = MouseParadigmConfig()
    config.continuous_cricket_confidence = True
    
    matrix, manifest = generate_features(
        pose_data=pose,
        video_width=514.0,
        video_fps=30.0,
        enabled_blocks=config.enabled_feature_blocks,
        config=config
    )
    
    assert matrix.shape == (num_frames, 45)
    vis_idx = manifest.feature_names.index("cricket_visible")
    vis_feature = matrix[:, vis_idx]
    
    # Frames 0..6 should match conf_values; frames 7..9 (NaN coords) must be 0.0
    np.testing.assert_allclose(vis_feature[:7], conf_values[:7], atol=1e-5)
    assert np.all(vis_feature[7:] == 0.0)
    print("Test 2: Continuous confidence feature test PASSED!")

def test_dlc_adapter_both_modes():
    """Validates DLCAdapter filtering under both continuous and legacy modes."""
    num_frames = 5
    scorer = "test_scorer"
    
    cols = pd.MultiIndex.from_tuples([
        (scorer, 'nose', 'x'), (scorer, 'nose', 'y'), (scorer, 'nose', 'likelihood'),
        (scorer, 'mid_body', 'x'), (scorer, 'mid_body', 'y'), (scorer, 'mid_body', 'likelihood'),
        (scorer, 'butt', 'x'), (scorer, 'butt', 'y'), (scorer, 'butt', 'likelihood'),
        (scorer, 'cricket', 'x'), (scorer, 'cricket', 'y'), (scorer, 'cricket', 'likelihood'),
    ])
    
    data = np.zeros((num_frames, len(cols)))
    df = pd.DataFrame(data, columns=cols)
    
    for bp in ['nose', 'mid_body', 'butt']:
        df[(scorer, bp, 'likelihood')] = 0.99
        df[(scorer, bp, 'x')] = 100.0
        df[(scorer, bp, 'y')] = 100.0
        
    # Cricket confidences: [0.90, 0.60, 0.40, 0.20, 0.05]
    df[(scorer, 'cricket', 'x')] = np.array([200.0, 210.0, 220.0, 230.0, 240.0])
    df[(scorer, 'cricket', 'y')] = np.array([200.0, 210.0, 220.0, 230.0, 240.0])
    df[(scorer, 'cricket', 'likelihood')] = np.array([0.90, 0.60, 0.40, 0.20, 0.05])
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as tmp:
        df.to_csv(tmp.name)
        tmp_path = tmp.name
        
    try:
        # 1. Mode A: Legacy mode (continuous_cricket_confidence = False)
        # Should filter at crick_th = 0.85 -> only frame 0 is retained
        config_legacy = MouseParadigmConfig()
        config_legacy.continuous_cricket_confidence = False
        config_legacy.thresholds.cricket_likelihood_thresh = 0.85
        config_legacy.thresholds.filter_poops = False
        
        adapter_legacy = DLCAdapter(config_legacy)
        pose_legacy = adapter_legacy.load(tmp_path, video_fps=30.0, interp_limit=0, filter_dead_crickets=False)
        
        assert not np.isnan(pose_legacy.cricket[0, 0]), "Frame 0 (0.90) must be retained in legacy mode"
        assert np.isnan(pose_legacy.cricket[1, 0]), "Frame 1 (0.60) must be NaN in legacy mode (>0.85 required)"
        assert np.isnan(pose_legacy.cricket[2, 0]), "Frame 2 (0.40) must be NaN in legacy mode"
        print("Test 3a: DLCAdapter legacy cutoff (0.85) PASSED!")

        # 2. Mode B: Continuous mode (continuous_cricket_confidence = True)
        # Should filter at noise floor coord_th = 0.30 -> frames 0, 1, 2 retained
        config_cont = MouseParadigmConfig()
        config_cont.continuous_cricket_confidence = True
        config_cont.thresholds.cricket_coord_thresh = 0.30
        config_cont.thresholds.cricket_likelihood_thresh = 0.85
        config_cont.thresholds.filter_poops = False
        
        adapter_cont = DLCAdapter(config_cont)
        pose_cont = adapter_cont.load(tmp_path, video_fps=30.0, interp_limit=0, filter_dead_crickets=False)
        
        assert not np.isnan(pose_cont.cricket[0, 0]), "Frame 0 (0.90) retained"
        assert not np.isnan(pose_cont.cricket[1, 0]), "Frame 1 (0.60) retained"
        assert not np.isnan(pose_cont.cricket[2, 0]), "Frame 2 (0.40) retained"
        assert np.isnan(pose_cont.cricket[3, 0]), "Frame 3 (0.20) NaN (<0.30)"
        assert np.isnan(pose_cont.cricket[4, 0]), "Frame 4 (0.05) NaN (<0.30)"
        
        assert abs(pose_cont.confidences['cricket'][1] - 0.60) < 1e-3
        assert abs(pose_cont.confidences['cricket'][2] - 0.40) < 1e-3
        assert pose_cont.confidences['cricket'][3] == 0.0
        print("Test 3b: DLCAdapter continuous noise floor (0.30) PASSED!")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_pipeline_metadata_autodetect():
    """Validates that MouseInferencePipeline auto-detects continuous_cricket_confidence from checkpoint."""
    from LSTM_model_training import HuntingBiLSTM
    dummy_model = HuntingBiLSTM(input_dim=45, hidden_dim=64, num_layers=1, num_classes=5)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_legacy:
        # Legacy checkpoint without 'continuous_cricket_confidence' key
        payload_legacy = {
            'target_fps': 30.0,
            'feat_dim': 45,
            'num_classes': 5,
            'hidden_dim': 64,
            'num_layers': 1,
            'model_state_dict': dummy_model.state_dict()
        }
        torch.save(payload_legacy, tmp_legacy.name)
        legacy_path = tmp_legacy.name

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_new:
        # New checkpoint with continuous_cricket_confidence = True
        payload_new = {
            'target_fps': 30.0,
            'feat_dim': 45,
            'num_classes': 5,
            'hidden_dim': 64,
            'num_layers': 1,
            'continuous_cricket_confidence': True,
            'model_state_dict': dummy_model.state_dict()
        }
        torch.save(payload_new, tmp_new.name)
        new_path = tmp_new.name

    try:
        # Legacy pipeline: auto-detects False
        pipe_legacy = MouseInferencePipeline(model_weights_path=legacy_path, device=torch.device('cpu'))
        pipe_legacy.init_model(input_dim=45)
        assert pipe_legacy.continuous_cricket_confidence is False
        assert pipe_legacy.config.continuous_cricket_confidence is False
        print("Test 4a: Pipeline legacy checkpoint auto-detection (False) PASSED!")

        # New pipeline: auto-detects True
        pipe_new = MouseInferencePipeline(model_weights_path=new_path, device=torch.device('cpu'))
        pipe_new.init_model(input_dim=45)
        assert pipe_new.continuous_cricket_confidence is True
        assert pipe_new.config.continuous_cricket_confidence is True
        print("Test 4b: Pipeline new checkpoint auto-detection (True) PASSED!")

        # Explicit override takes priority
        pipe_override = MouseInferencePipeline(model_weights_path=new_path, continuous_cricket_confidence=False, device=torch.device('cpu'))
        pipe_override.init_model(input_dim=45)
        assert pipe_override.continuous_cricket_confidence is False
        assert pipe_override.config.continuous_cricket_confidence is False
        print("Test 4c: Pipeline explicit override PASSED!")
    finally:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
        if os.path.exists(new_path):
            os.remove(new_path)

if __name__ == "__main__":
    test_legacy_boolean_mode()
    test_continuous_mode_feature()
    test_dlc_adapter_both_modes()
    test_pipeline_metadata_autodetect()
    print("\n==============================================")
    print("ALL TESTS PASSED SUCCESSFULLY (100% COMPLIANT)!")
    print("==============================================")
