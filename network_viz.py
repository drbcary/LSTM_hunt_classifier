import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib

# CRITICAL: Use 'Agg' backend for stable video generation
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import umap
import imageio.v3 as iio
from sklearn.preprocessing import MinMaxScaler
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. Imports & Configuration
# ==========================================

INPUT_DIM = 11
HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_CLASSES = 4 
PERCENT_BG = 1.0 # 100% of features for background cloud

# User-specific paths
DLC_FILE = r"Z:\LucasWahl\Exp 8 - Re-do of naive Wt and naive Scn2a data\hunt_crop\scn2a\LW134-hunt-1-03112025102959_huntPerfDLC_resnet50_Lucas_HuntObsMar6shuffle1_2430000.csv"
LABEL_FILE = r"Z:\LucasWahl\Excel files\Exp 8 - naive wt vs naive scn2a\Exp 8 - naive wt vs naive scn2a\test_analyzed\scn2a\LW134-Hunt-1_analyzed.xlsx"
MODEL_PATH = 'BGadd_feat11_dim256_lay3_ep50_valf1=0.8865.pth'

# ==========================================
# 2. Model Definitions
# ==========================================

class HuntingBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(output)
        return logits

class TransparentStackedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True))
        for _ in range(1, num_layers):
            self.layers.append(nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True))

    def forward(self, x):
        layer_outputs = []
        current_input = x
        for layer in self.layers:
            out, _ = layer(current_input)
            layer_outputs.append(out)
            current_input = out
        return current_input, layer_outputs

def transfer_weights(source_model, target_model):
    print("Migrating weights from HuntingBiLSTM to Visualizer...")
    src_lstm = source_model.lstm 
    with torch.no_grad():
        for i in range(target_model.num_layers):
            target_layer = target_model.layers[i]
            target_layer.weight_ih_l0.copy_(getattr(src_lstm, f'weight_ih_l{i}'))
            target_layer.weight_hh_l0.copy_(getattr(src_lstm, f'weight_hh_l{i}'))
            target_layer.weight_ih_l0_reverse.copy_(getattr(src_lstm, f'weight_ih_l{i}_reverse'))
            target_layer.weight_hh_l0_reverse.copy_(getattr(src_lstm, f'weight_hh_l{i}_reverse'))
            target_layer.bias_ih_l0.copy_(getattr(src_lstm, f'bias_ih_l{i}'))
            target_layer.bias_hh_l0.copy_(getattr(src_lstm, f'bias_hh_l{i}'))
            target_layer.bias_ih_l0_reverse.copy_(getattr(src_lstm, f'bias_ih_l{i}_reverse'))
            target_layer.bias_hh_l0_reverse.copy_(getattr(src_lstm, f'bias_hh_l{i}_reverse'))
    print("Weight transfer complete.")

# ==========================================
# 3. Data Loading & Inference Pipeline
# ==========================================

features_all = None
preds_all = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from RatHunt_Inference import run_inference_on_video
    import dataset_utils
    print("Successfully imported RatHunt_Inference modules.")
    
    print(f"Loading Model from {MODEL_PATH}...")
    pretrained_model = HuntingBiLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(MODEL_PATH):
        pretrained_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Warning: Model path {MODEL_PATH} not found. Initializing random weights for demo.")

    print(f"Processing DLC File: {DLC_FILE}...")
    if os.path.exists(DLC_FILE):
        probs, preds = run_inference_on_video(DLC_FILE, pretrained_model, device)
        preds_all = preds.cpu().numpy() if torch.is_tensor(preds) else preds
        
        video_width, _, video_fps = dataset_utils.get_video_params(DLC_FILE)
        res = dataset_utils.load_and_process_dlc(DLC_FILE, video_fps)
        rh, rm, rb, rt, cp = res
        features = dataset_utils.generate_features(rh, rm, rb, rt, cp, video_width, video_fps)
        
        if torch.is_tensor(features):
            features_all = features.cpu().numpy()
        else:
            features_all = features
            
        print(f"Data Loaded. Shape: {features_all.shape}, Labels: {preds_all.shape}")
        
    else:
        raise FileNotFoundError("DLC File not found.")

except (ImportError, FileNotFoundError) as e:
    print(f"\nExample Pipeline Failed: {e}")
    print("falling back to DUMMY DATA Generation for demonstration...")
    
    total_frames = 30000 
    features_all = np.zeros((total_frames, INPUT_DIM), dtype=np.float32)
    t = np.linspace(0, 100, total_frames)
    for i in range(INPUT_DIM):
        freq = 0.5 + (i % 3) * 0.2
        features_all[:, i] = np.sin(t * freq) + np.random.normal(0, 0.1, total_frames)
    
    pretrained_model = HuntingBiLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    pretrained_model.eval()
    
    dummy_tensor = torch.FloatTensor(features_all).unsqueeze(0).to(device)
    dummy_lens = torch.tensor([total_frames]).cpu()
    with torch.no_grad():
        logits = pretrained_model(dummy_tensor, dummy_lens)
        preds_all = torch.argmax(logits, dim=2).cpu().numpy().squeeze()

# ==========================================
# 4. Visualization Model Setup
# ==========================================

visualizer_model = TransparentStackedBiLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to('cpu')
pretrained_model = pretrained_model.to('cpu')
visualizer_model.eval()
transfer_weights(pretrained_model, visualizer_model)

# ==========================================
# 5. Internal State Extraction
# ==========================================
print("Running Inspector on Feature Data...")

features_tensor = torch.FloatTensor(features_all).unsqueeze(0) 

with torch.no_grad():
    final_out, all_layers_activity = visualizer_model(features_tensor)
    
    print("\n--- Verifying Transparent Model Integrity ---")
    transp_logits = pretrained_model.fc(final_out)
    
    # NEW: Get Probabilities for Output Layer Visualization
    transp_probs = torch.softmax(transp_logits, dim=2).cpu().numpy().squeeze() # [Frames, 4]
    
    transp_preds = torch.argmax(transp_logits, dim=2).cpu().numpy().squeeze()
    
    orig_flat = preds_all.flatten()
    new_flat = transp_preds.flatten()
    
    match_rate = np.mean(orig_flat == new_flat)
    print(f"Match Rate between Original and Transparent Model: {match_rate * 100:.2f}%")
    print("---------------------------------------------\n")

brain_states = torch.cat(all_layers_activity, dim=2).squeeze(0).numpy()

# ==========================================
# ANIMATION WINDOW DEFINITION
# ==========================================
start_frame = 26140
end_frame = 26680

start_idx = max(0, start_frame)
end_idx = min(end_frame, len(features_all))

if start_idx >= end_idx:
    print(f"Error: Start frame {start_idx} is >= End frame {end_idx}. Using default first 300 frames.")
    start_idx = 0
    end_idx = 300

print(f"Animating sequence from frame {start_idx} to {end_idx} ({end_idx - start_idx} frames)")

anim_brain_states = brain_states[start_idx:end_idx] 
anim_layer_activity = [l[:, start_idx:end_idx, :] for l in all_layers_activity]
anim_probs = transp_probs[start_idx:end_idx] # Output layer probabilities

# ==========================================
# 6. Connection Web Processing
# ==========================================
def extract_strongest_connections(layer_module, top_k=150):
    weights = layer_module.weight_ih_l0.detach().cpu().numpy()
    chunk_size = weights.shape[0] // 4
    w_avg = np.zeros((chunk_size, weights.shape[1]))
    for i in range(4):
        w_avg += np.abs(weights[i*chunk_size : (i+1)*chunk_size, :])
    w_avg /= 4.0
    flat_indices = np.argsort(w_avg.ravel())[-top_k:]
    dst_indices, src_indices = np.unravel_index(flat_indices, w_avg.shape)
    magnitudes = w_avg[dst_indices, src_indices]
    magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-6)
    return src_indices, dst_indices, magnitudes

# NEW: Function for FC layer connections
def extract_fc_connections(fc_module, top_k=100):
    # Weights shape: [Out_Dim, In_Dim] -> [4, 512]
    weights = fc_module.weight.detach().cpu().numpy()
    
    # We use magnitude for SELECTION
    w_abs = np.abs(weights)
    flat_indices = np.argsort(w_abs.ravel())[-top_k:]
    dst_indices, src_indices = np.unravel_index(flat_indices, w_abs.shape)
    
    # But we return RAW weights to preserve SIGN (+/-)
    raw_weights = weights[dst_indices, src_indices]

    # Map dst_indices to horizontal position of class segment center for visualization
    hidden_width = weights.shape[1]
    seg_len = hidden_width // 4
    dst_x_coords = dst_indices * seg_len + seg_len // 2
    
    return src_indices, dst_x_coords, raw_weights

print("Extracting Synaptic Web...")
src_01, dst_01, w_01 = extract_strongest_connections(visualizer_model.layers[1], top_k=150)
src_12, dst_12, w_12 = extract_strongest_connections(visualizer_model.layers[2], top_k=150)
# Extract L3 -> Output connections
src_2out, dst_2out_x, w_2out_raw = extract_fc_connections(pretrained_model.fc, top_k=150) # Increased K slightly

# Normalize magnitudes for line width (using abs)
w_2out_abs = np.abs(w_2out_raw)
w_2out_norm = (w_2out_abs - w_2out_abs.min()) / (w_2out_abs.max() - w_2out_abs.min() + 1e-6)

# ==========================================
# 7. UMAP Projection
# ==========================================
print("Computing UMAP Projection...")

num_samples = int(len(features_all) * PERCENT_BG)
indices = np.random.choice(len(features_all), num_samples, replace=False)
bg_states = brain_states[indices]
bg_labels = preds_all[indices]

traj_states = anim_brain_states

# Reverted to n_neighbors=30
reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
bg_embedding = reducer.fit_transform(bg_states)
traj_embedding = reducer.transform(traj_states)

all_coords = np.vstack([bg_embedding, traj_embedding])
mean = np.mean(all_coords, axis=0)
scale = 10.0 / (np.max(np.abs(all_coords - mean)) + 1e-5)

bg_embedding = (bg_embedding - mean) * scale
traj_embedding = (traj_embedding - mean) * scale

# ==========================================
# 8. Visualization Setup
# ==========================================
print("Setting up Graphics...")

# Prepare Heatmap Data (Masked)
STRIP_HEIGHT = 50
GAP_HEIGHT = 60
heatmap_frames = []
seq_length = end_idx - start_idx
hidden_width = anim_layer_activity[0].shape[2] # Should be 512

for t in range(seq_length):
    strips = []
    
    # --- OUTPUT LAYER (Top) ---
    probs = anim_probs[t] # [4]
    output_strip = np.zeros(hidden_width)
    seg_len = hidden_width // 4
    for c in range(4):
        output_strip[c*seg_len : (c+1)*seg_len] = probs[c]
    
    strips.append(np.tile(output_strip, (STRIP_HEIGHT, 1)))
    strips.append(np.zeros((GAP_HEIGHT, hidden_width))) # Gap
    
    # --- HIDDEN LAYERS ---
    l2 = anim_layer_activity[2][0, t, :].numpy()
    strips.append(np.tile(l2, (STRIP_HEIGHT, 1)))
    strips.append(np.zeros((GAP_HEIGHT, l2.shape[0])))
    l1 = anim_layer_activity[1][0, t, :].numpy()
    strips.append(np.tile(l1, (STRIP_HEIGHT, 1)))
    strips.append(np.zeros((GAP_HEIGHT, l1.shape[0])))
    l0 = anim_layer_activity[0][0, t, :].numpy()
    strips.append(np.tile(l0, (STRIP_HEIGHT, 1)))
    
    heatmap_frames.append(np.vstack(strips))

heatmap_data_raw = np.array(heatmap_frames)

# Updated Mask Logic for 4 layers (3 gaps)
total_h = heatmap_data_raw.shape[1]
mask = np.zeros((total_h, heatmap_data_raw.shape[2]), dtype=bool)

# Gap 1 (Output -> L2)
mask[STRIP_HEIGHT : STRIP_HEIGHT+GAP_HEIGHT, :] = True
# Gap 2 (L2 -> L1)
mask[STRIP_HEIGHT*2+GAP_HEIGHT : STRIP_HEIGHT*2+GAP_HEIGHT*2, :] = True
# Gap 3 (L1 -> L0)
mask[STRIP_HEIGHT*3+GAP_HEIGHT*2 : STRIP_HEIGHT*3+GAP_HEIGHT*3, :] = True

full_mask = np.broadcast_to(mask, heatmap_data_raw.shape)

scaler = MinMaxScaler()
heatmap_flat = scaler.fit_transform(heatmap_data_raw.reshape(-1, 1))
heatmap_data_masked = np.ma.masked_array(heatmap_flat.reshape(heatmap_data_raw.shape), mask=full_mask)

# Center Y-Coordinates for Labels/Lines
y_out_center = STRIP_HEIGHT // 2
y_l2_center = STRIP_HEIGHT + GAP_HEIGHT + (STRIP_HEIGHT // 2)
y_l1_center = (STRIP_HEIGHT + GAP_HEIGHT) * 2 + (STRIP_HEIGHT // 2)
y_l0_center = (STRIP_HEIGHT + GAP_HEIGHT) * 3 + (STRIP_HEIGHT // 2)

# Line Edges
y_l0_top = y_l0_center - STRIP_HEIGHT // 2
y_l1_bottom = y_l1_center + STRIP_HEIGHT // 2
y_l1_top = y_l1_center - STRIP_HEIGHT // 2
y_l2_bottom = y_l2_center + STRIP_HEIGHT // 2
y_l2_top = y_l2_center - STRIP_HEIGHT // 2
y_out_bottom = y_out_center + STRIP_HEIGHT // 2

# Matplotlib Setup
plt.style.use('dark_background')
# ADJUSTED DIMENSIONS: Width 16 (20 - 20%), Height 14.08 (10 + 40%, made divisible by 16)
fig = plt.figure(figsize=(16, 14.08)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8], wspace=0.05)

# --- Panel 1: Neural Scan ---
ax_cortex = fig.add_subplot(gs[0])
ax_cortex.axis('off')
ax_cortex.set_title("Neural Synaptic Web", color='white', pad=20, fontsize=16)

im_display = ax_cortex.imshow(heatmap_data_masked[0], cmap='inferno', aspect='auto', vmin=0, vmax=1, interpolation='nearest')

# Labels
label_x_pos = -20 
ax_cortex.text(label_x_pos, y_out_center - STRIP_HEIGHT//2 - 10, "Output Layer", color='white', fontsize=10, fontweight='bold', ha='right', va='bottom')
ax_cortex.text(label_x_pos, y_l2_center - STRIP_HEIGHT//2 - 10, "Layer 3 (High)", color='white', fontsize=10, fontweight='bold', ha='right', va='bottom')
ax_cortex.text(label_x_pos, y_l1_center - STRIP_HEIGHT//2 - 10, "Layer 2 (Mid)", color='white', fontsize=10, fontweight='bold', ha='right', va='bottom')
ax_cortex.text(label_x_pos, y_l0_center - STRIP_HEIGHT//2 - 10, "Layer 1 (Input)", color='white', fontsize=10, fontweight='bold', ha='right', va='bottom')

class_names = ['Background', 'Cricket', 'Pursuit', 'Eating']
seg_len = hidden_width // 4
for i, name in enumerate(class_names):
    x_pos = i * seg_len + seg_len // 2
    ax_cortex.text(x_pos, y_out_center, name, color='black', fontsize=9, fontweight='bold', ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))

# Static Lines
def create_segments(src_idx, dst_idx, y_src, y_dst):
    return [[(s, y_src), (d, y_dst)] for s, d in zip(src_idx, dst_idx)]

seg_01 = create_segments(src_01, dst_01, y_l0_top, y_l1_bottom)
lc_01 = LineCollection(seg_01, colors='#008080', linewidths=w_01 * 2) 
ax_cortex.add_collection(lc_01)

seg_12 = create_segments(src_12, dst_12, y_l1_top, y_l2_bottom)
lc_12 = LineCollection(seg_12, colors='#008080', linewidths=w_12 * 2)
ax_cortex.add_collection(lc_12)

# L2 -> Output Lines
seg_2out = create_segments(src_2out, dst_2out_x, y_l2_top, y_out_bottom)
lc_2out = LineCollection(seg_2out, colors='#FFD700', linewidths=w_2out_norm * 3) # Init width with norm
ax_cortex.add_collection(lc_2out)

# --- Panel 2: UMAP Trajectory ---
ax_mind = fig.add_subplot(gs[1], projection='3d')
ax_mind.set_title("Network State Space", color='white', pad=0, fontsize=16)
ax_mind.set_axis_off()

classes = ['Background', 'Cricket Present', 'Pursuit', 'Eating']
custom_colors = ['lightgray', 'orange', 'red', 'green']
cmap = ListedColormap(custom_colors)
colors = cmap(bg_labels)

# Increased s to 8 and alpha to 0.5
ax_mind.scatter(bg_embedding[:,0], bg_embedding[:,1], bg_embedding[:,2], c=colors, s=8, alpha=0.5, depthshade=True)

line_collection = Line3DCollection([], linewidths=8, zorder=10) 
ax_mind.add_collection(line_collection)
scatter_head = ax_mind.scatter([], [], [], color='white', s=400, edgecolors='black', zorder=11) 

ax_mind.set_box_aspect((1, 1, 1))
limit = 6 
ax_mind.set_xlim(-limit, limit); ax_mind.set_ylim(-limit, limit); ax_mind.set_zlim(-limit, limit)

# ==========================================
# 9. Render Loop
# ==========================================
frames = []
num_frames = len(heatmap_data_masked)
print(f"Rendering {num_frames} frames...")

for i in range(num_frames):
    # Update Heatmap
    im_display.set_data(heatmap_data_masked[i])
    
    # Update Lines (L0->L1, L1->L2)
    act_0 = np.clip(np.abs(anim_layer_activity[0][0, i, src_01].numpy()), 0, 1)
    c_01 = np.zeros((len(src_01), 4))
    c_01[:, :3] = [0, 0.9, 0.9] 
    c_01[:, 3] = act_0 * 0.8 + 0.1
    lc_01.set_color(c_01)
    
    act_1 = np.clip(np.abs(anim_layer_activity[1][0, i, src_12].numpy()), 0, 1)
    c_12 = np.zeros((len(src_12), 4))
    c_12[:, :3] = [0, 0.9, 0.9]
    c_12[:, 3] = act_1 * 0.8 + 0.1
    lc_12.set_color(c_12)

    # NEW: Update L2->Output Lines (Color by Sign)
    act_2 = np.clip(np.abs(anim_layer_activity[2][0, i, src_2out].numpy()), 0, 1)
    
    # Initialize color array
    c_2out = np.zeros((len(src_2out), 4))
    
    # Mask for Positive (Excitatory) Weights
    pos_mask = w_2out_raw > 0
    # Mask for Negative (Inhibitory) Weights
    neg_mask = ~pos_mask
    
    # 1. POSITIVE -> Gold
    c_2out[pos_mask, :3] = [1.0, 0.84, 0.0] # Gold
    c_2out[pos_mask, 3] = act_2[pos_mask] * 0.9 + 0.1 # High visibility
    
    # 2. NEGATIVE -> Faint Blue/Grey (Inhibitory)
    # Increased visibility here (* 0.6 + 0.2)
    c_2out[neg_mask, :3] = [0.2, 0.4, 0.6] # Steel Blue
    c_2out[neg_mask, 3] = act_2[neg_mask] * 0.6 + 0.2 # HIGH visibility

    lc_2out.set_color(c_2out)
    
    # Update Trajectory
    if i > 1:
        pts = traj_embedding[:i+1].reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        
        trail_dur = 100
        alphas = np.linspace(0, 1, len(segs))
        if len(alphas) > trail_dur:
            alphas[:len(alphas)-trail_dur] = 0
            alphas[len(alphas)-trail_dur:] = np.linspace(0, 1, trail_dur)
            
        colors_line = np.zeros((len(segs), 4))
        colors_line[:, 1] = 1.0; colors_line[:, 2] = 1.0 
        colors_line[:, 3] = alphas
        
        line_collection.set_segments(segs)
        line_collection.set_color(colors_line)
        
        head = traj_embedding[i]
        scatter_head._offsets3d = ([head[0]], [head[1]], [head[2]])
        
    ax_mind.view_init(elev=20, azim=(i * 0.5))
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf).copy()
    if img.shape[2] == 4: img = img[:,:,:3]
    frames.append(img)
    
    if i % 10 == 0: print(f"Frame {i}/{num_frames}")

plt.close(fig)
iio.imwrite("network_activity.mp4", frames, fps=30)
print("Done. Saved to network_activity.mp4")