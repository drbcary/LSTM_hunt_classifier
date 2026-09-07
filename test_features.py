import numpy as np

def calculate_bearing(start_pt, end_pt):
    return np.arctan2(end_pt[:, 1] - start_pt[:, 1], end_pt[:, 0] - start_pt[:, 0])

def calculate_angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps):
    def dist(p1, p2): return np.linalg.norm(p1 - p2, axis=1)
    def get_derivs(pos_array):
        vel_vec = np.diff(pos_array, axis=0, prepend=pos_array[0:1]) * video_fps
        speed = np.linalg.norm(vel_vec, axis=1) / video_width
        acc_vec = np.diff(vel_vec, axis=0, prepend=vel_vec[0:1]) * video_fps
        accel = np.linalg.norm(acc_vec, axis=1) / video_width
        return speed, accel

    d_head_cricket = dist(rat_head, cricket_pos) / video_width
    d_head_mid = dist(rat_head, rat_mid) / video_width
    d_mid_tail = dist(rat_mid, rat_tail) / video_width
    d_head_tail = dist(rat_head, rat_tail) / video_width
    
    angle_spine = calculate_angle_3pts(rat_head, rat_mid, rat_tail) / 180.0
    rat_orientation = calculate_bearing(rat_mid, rat_head)
    target_vector = calculate_bearing(rat_head, cricket_pos)
    angle_diff = target_vector - rat_orientation
    angle_diff_raw = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    angle_diff = angle_diff_raw / np.pi
    
    rat_speed, rat_accel = get_derivs(rat_head)
    cricket_speed, cricket_accel = get_derivs(cricket_pos)
    cricket_visible = np.where(np.isnan(d_head_cricket), 0.0, 1.0)
    
    # === NEW FEATURES ===
    
    # 1. Closing Velocity & Inverse TTC
    closing_velocity = -np.diff(d_head_cricket, prepend=d_head_cricket[0:1]) * video_fps
    inverse_ttc = np.clip(closing_velocity / (d_head_cricket + 1e-6), -100, 100) 

    # 2. Angular Velocity & Aim
    def angular_diff(angles):
        d_angles = np.diff(angles, prepend=angles[0:1])
        return (d_angles + np.pi) % (2 * np.pi) - np.pi

    rat_turn_rate = np.abs(angular_diff(rat_orientation)) * video_fps / np.pi
    aim_correction_rate = np.abs(angular_diff(angle_diff_raw)) * video_fps / np.pi

    # 3. Lunging & Egocentric Mapping
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

    # 4. Body Posture & Tail Whip
    body_compression = d_head_tail / (d_head_mid + d_mid_tail + 1e-6)
    compression_rate = np.diff(body_compression, prepend=body_compression[0:1]) * video_fps

    angle_tail = calculate_angle_3pts(rat_mid, rat_butt, rat_tail) / 180.0
    tail_angular_vel = np.abs(np.diff(angle_tail, prepend=angle_tail[0:1])) * video_fps

    # 5. Cricket Evasion (Jitter)
    c_vel = np.diff(cricket_pos, axis=0, prepend=cricket_pos[0:1])
    c_bearing = np.arctan2(c_vel[:, 1], c_vel[:, 0])
    c_jitter = np.abs(angular_diff(c_bearing)) / np.pi
    
    feature_stack = np.column_stack([
        # Original 11 features
        d_head_cricket, d_head_mid, d_mid_tail, d_head_tail,
        angle_spine, angle_diff, rat_speed, rat_accel,
        cricket_speed, cricket_accel, cricket_visible,
        
        # New 11 features
        closing_velocity, inverse_ttc, rat_turn_rate, aim_correction_rate,
        lunge_speed, ego_forward, ego_lateral, body_compression,
        compression_rate, tail_angular_vel, c_jitter
    ])
    return np.nan_to_num(feature_stack, nan=0.0)

# Dummy test
num_frames = 10
rat_head = np.random.rand(num_frames, 2) * 100
rat_mid = np.random.rand(num_frames, 2) * 100
rat_butt = np.random.rand(num_frames, 2) * 100
rat_tail = np.random.rand(num_frames, 2) * 100
cricket_pos = np.random.rand(num_frames, 2) * 100
video_width = 1920
video_fps = 30

features = generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)
print("Shape of features:", features.shape)
assert features.shape == (10, 22)
print("Success!")
