# main.py — Feature normalization only (no target normalization)
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os, re
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import graph
import model as gnn_model

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CURRENT_DIR = os.getcwd()
DATA_PATH = os.path.join(CURRENT_DIR, "data_sets")
print("Notebook started")

# --- Extract assembly/timestep info ---
def get_data_info(data_path):
    assembly_ids = set()
    timesteps = set()
    pattern = re.compile(r"(\d+)_(\d+)_ball_NPMNCF\.tab")
    for fname in os.listdir(data_path):
        match = pattern.match(fname)
        if match:
            assembly_ids.add(int(match.group(1)))
            timesteps.add(int(match.group(2)))
    return sorted(list(assembly_ids)), len(timesteps)

ASSEMBLY_IDS, TIMESTEPS = get_data_info(DATA_PATH)
ASSEMBLY_IDS_TO_PROCESS = ASSEMBLY_IDS[:100]
print(f"Discovered {len(ASSEMBLY_IDS)} assemblies with {TIMESTEPS} timesteps each.")

# --- Logging setup ---
writer = SummaryWriter(f"runs/gnn_training_experiment_{datetime.now().strftime('%d_%m_%y_%H-%M-%S.%f')}")
graph.set_global_paths_and_device(DATA_PATH, DEVICE, TIMESTEPS)
gnn_model.set_device_for_model(DEVICE)

# --- Load/Process Graphs ---
PROCESSED_DATA_FILE = "processed_graphs_and_targets.pt"
if os.path.exists(PROCESSED_DATA_FILE):
    print(f"Loading graphs and targets from {PROCESSED_DATA_FILE}...")
    graphs, targets = torch.load(PROCESSED_DATA_FILE)
else:
    print("Creating graphs...")
    TIMESTEPS = 81
    graphs, targets = graph.create_all_graphs(assembly_ids=ASSEMBLY_IDS_TO_PROCESS)
    torch.save((graphs, targets), PROCESSED_DATA_FILE)
    print("Data saved successfully.")

# --- Train/Test Split ---
NUM_TRAIN = 800
NUM_TEST = 800
total_graphs = len(graphs)
indices = np.random.permutation(total_graphs)
train_indices = indices[:NUM_TRAIN]
test_indices = indices[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
train_graph_dicts = [graphs[i] for i in train_indices]
train_targets = [targets[i] for i in train_indices]
test_graph_dicts = [graphs[i] for i in test_indices]
test_targets = [targets[i] for i in test_indices]

# --- Feature Normalization (only inputs, not targets) ---
all_nodes_for_norm = torch.cat([g["nodes"].cpu() for g in train_graph_dicts], dim=0)
node_mean = all_nodes_for_norm.mean(dim=0, keepdim=True)
node_std = all_nodes_for_norm.std(dim=0, keepdim=True) + 1e-8

all_edges_for_norm = torch.cat([g["edges"].cpu() for g in train_graph_dicts if g["edges"].numel() > 0], dim=0)
edge_mean = all_edges_for_norm.mean(dim=0, keepdim=True)
edge_std = all_edges_for_norm.std(dim=0, keepdim=True) + 1e-8

print("✅ Input feature normalization applied (nodes + edges). Targets remain raw.")

# --- Hyperparameters (per paper) ---
num_epochs = 100
batch_size = 1
num_processing_steps = 7
clip_norm = 5.0
learning_rate = 1e-3

# --- Model Setup ---
model = gnn_model.EncodeProcessDecode(
    node_input_dim=2,
    edge_input_dim=3,
    node_output_size=1,
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

num_processing_steps_tensor_for_model = torch.tensor(num_processing_steps, dtype=torch.int32).to(DEVICE)

# --- Training history tracking ---
l1_train_hist, l2_train_hist, rho_train_hist = [], [], []
l1_test_hist, l2_test_hist, rho_test_hist = [], [], []

print("🚀 Starting training (feature-normalized, raw targets)...")
for epoch in range(num_epochs):
    model.train()
    l1s_train, l2s_train, rhos_train = [], [], []

    # ---- TRAIN PHASE ----
    for i in range(0, len(train_graph_dicts), batch_size):
        batch_graphs = train_graph_dicts[i:i + batch_size]
        batch_targets = train_targets[i:i + batch_size]
        if not batch_graphs:
            continue

        # Normalize only node and edge features
        normalized_graphs = []
        for g in batch_graphs:
            g_norm = g.copy()
            g_norm["nodes"] = ((g_norm["nodes"].cpu() - node_mean) / node_std).to(DEVICE)
            if g_norm["edges"].numel() > 0:
                g_norm["edges"] = ((g_norm["edges"].cpu() - edge_mean) / edge_std).to(DEVICE)
            normalized_graphs.append(g_norm)

        graphs_tuple = graph.data_dicts_to_graphs_tuple_pytorch(normalized_graphs)
        targets_concat = torch.cat(batch_targets, dim=0).to(DEVICE)  # raw targets (not normalized)

        optimizer.zero_grad()
        outputs = model(graphs_tuple, num_processing_steps_tensor_for_model)
        preds = outputs[-1]["nodes"]

        l1_loss_val = F.l1_loss(preds, targets_concat)
        l2_loss_val = F.mse_loss(preds, targets_concat)
        rho_val = graph.pearson_corr_pytorch(preds, targets_concat)

        l2_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        l1s_train.append(l1_loss_val.item())
        l2s_train.append(l2_loss_val.item())
        rhos_train.append(rho_val.item())

    L1_train, L2_train, rho_train = np.mean(l1s_train), np.mean(l2s_train), np.mean(rhos_train)
    l1_train_hist.append(L1_train)
    l2_train_hist.append(L2_train)
    rho_train_hist.append(rho_train)

    # ---- TEST PHASE ----
    model.eval()
    l1s_test, l2s_test, rhos_test = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_graph_dicts), batch_size):
            batch_graphs = test_graph_dicts[i:i + batch_size]
            batch_targets = test_targets[i:i + batch_size]
            if not batch_graphs:
                continue

            normalized_graphs = []
            for g in batch_graphs:
                g_norm = g.copy()
                g_norm["nodes"] = ((g_norm["nodes"].cpu() - node_mean) / node_std).to(DEVICE)
                if g_norm["edges"].numel() > 0:
                    g_norm["edges"] = ((g_norm["edges"].cpu() - edge_mean) / edge_std).to(DEVICE)
                normalized_graphs.append(g_norm)

            graphs_tuple = graph.data_dicts_to_graphs_tuple_pytorch(normalized_graphs)
            targets_concat = torch.cat(batch_targets, dim=0).to(DEVICE)  # raw targets

            outputs = model(graphs_tuple, num_processing_steps_tensor_for_model)
            preds_test = outputs[-1]["nodes"]

            l1s_test.append(F.l1_loss(preds_test, targets_concat).item())
            l2s_test.append(F.mse_loss(preds_test, targets_concat).item())
            rhos_test.append(graph.pearson_corr_pytorch(preds_test, targets_concat).item())

    L1_test, L2_test, rho_test = np.mean(l1s_test), np.mean(l2s_test), np.mean(rhos_test)
    l1_test_hist.append(L1_test)
    l2_test_hist.append(L2_test)
    rho_test_hist.append(rho_test)

    print(f"Epoch {epoch+1:03d}: "
          f"L1_train={L1_train:.4f}, L2_train={L2_train:.4f}, rho_train={rho_train:.4f} | "
          f"L1_test={L1_test:.4f}, L2_test={L2_test:.4f}, rho_test={rho_test:.4f}")

print(f"✅ Training completed. Final rho_test = {rho_test:.4f}")

# ---------------------------------------------------------
# 📊 Visualization
# ---------------------------------------------------------
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, l1_train_hist, label='Train L1', linewidth=2)
plt.plot(epochs, l1_test_hist, label='Test L1', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.title('Train vs Test L1 (Raw Targets)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 3, 2)
plt.plot(epochs, l2_train_hist, label='Train L2', linewidth=2)
plt.plot(epochs, l2_test_hist, label='Test L2', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('L2 Loss')
plt.title('Train vs Test L2 (Raw Targets)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 3, 3)
plt.plot(epochs, rho_train_hist, label='Train ρ', linewidth=2)
plt.plot(epochs, rho_test_hist, label='Test ρ', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Pearson ρ')
plt.title('Train vs Test Correlation')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("training_results_features_norm_only.png", dpi=300)
plt.show()

print("✅ Results saved as 'training_results_features_norm_only.png'")
