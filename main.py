# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import re # Import the regular expression module
from collections import deque
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import graph
import model as gnn_model
# import vis

# --- Global Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CURRENT_DIR = os.getcwd()
DATA_PATH = os.path.join(CURRENT_DIR, 'Trials_1-135')
print("Notebook started")

# Dynamically determine ASSEMBLY_IDS and TIMESTEPS from file names
def get_data_info(data_path):
    assembly_ids = set()
    timesteps = set()
    
    # Regex to find assembly ID and timestep from file names like "134_18_contact_end1.tab"
    pattern = re.compile(r'(\d+)_(\d+)_ball_NPMNCF\.tab')

    for fname in os.listdir(data_path):
        match = pattern.match(fname)
        if match:
            assembly_ids.add(int(match.group(1)))
            timesteps.add(int(match.group(2)))
    
    return sorted(list(assembly_ids)), len(timesteps)

ASSEMBLY_IDS, TIMESTEPS = get_data_info(DATA_PATH)
ASSEMBLY_IDS_TO_PROCESS = ASSEMBLY_IDS[:100]
print(f"Discovered {len(ASSEMBLY_IDS)} assemblies with {TIMESTEPS} timesteps each.")

from datetime import datetime
writer = SummaryWriter(f'runs/gnn_training_experiment_{datetime.now().strftime("%d_%m_%y_%H-%M-%S.%f")}')

graph.set_global_paths_and_device(DATA_PATH, DEVICE, TIMESTEPS)
gnn_model.set_device_for_model(DEVICE)


# --- Data Loading with Save/Load Mechanism ---
PROCESSED_DATA_FILE = 'processed_graphs_and_targets.pt'
ASSEMBLY_IDS, _ = get_data_info(DATA_PATH)

if os.path.exists(PROCESSED_DATA_FILE):
    print(f"Loading graphs and targets from {PROCESSED_DATA_FILE}...")
    graphs, targets = torch.load(PROCESSED_DATA_FILE)
    print("Data loaded successfully.")
else:
    print(f"File {PROCESSED_DATA_FILE} not found. Creating graphs...")
    TIMESTEPS = 21 # Assuming each assembly has 21 timesteps
    graphs, targets = graph.create_all_graphs(assembly_ids=ASSEMBLY_IDS_TO_PROCESS)
    
    print("Saving processed graphs and targets...")
    torch.save((graphs, targets), PROCESSED_DATA_FILE)
    print("Data saved successfully for future use.")

# --- Data Splitting with Fixed Counts ---
NUM_TRAIN = 1500
NUM_TEST = 400

total_graphs = len(graphs)

# Ensure you have enough graphs for the requested split
if total_graphs < NUM_TRAIN + NUM_TEST:
    raise ValueError(f"Not enough total graphs ({total_graphs}) for the requested split of {NUM_TRAIN} train and {NUM_TEST} test.")

# Shuffle the data to ensure random splits
indices = np.random.permutation(total_graphs)
# indices= np.arange(total_graphs)

# Split the indices into train and test sets
train_indices = indices[:NUM_TRAIN]
test_indices = indices[NUM_TRAIN:NUM_TRAIN + NUM_TEST]

# Create data splits using the shuffled indices
train_graph_dicts = [graphs[i] for i in train_indices]
train_targets = [targets[i] for i in train_indices]
test_graph_dicts = [graphs[i] for i in test_indices]
test_targets = [targets[i] for i in test_indices]

# --- Normalization Statistics Calculation ---
train_graphs_for_stats = train_graph_dicts
all_nodes_for_norm = torch.cat([g["nodes"].cpu() for g in train_graphs_for_stats], dim=0)
all_edges_for_norm_list = [g["edges"].cpu() for g in train_graphs_for_stats if g["edges"].numel() > 0]

node_mean = all_nodes_for_norm.mean(dim=0, keepdim=True)
node_std = all_nodes_for_norm.std(dim=0, keepdim=True) + 1e-8

edge_mean = None
edge_std = None
if len(all_edges_for_norm_list) > 0:
    all_edges_for_norm = torch.cat(all_edges_for_norm_list, dim=0)
    edge_mean = all_edges_for_norm.mean(dim=0, keepdim=True)
    edge_std = all_edges_for_norm.std(dim=0, keepdim=True) + 1e-8

print("Normalization statistics calculated from training graphs.")

# --- Hyperparameters ---
num_epochs = 100
batch_size = 1
num_processing_steps = 5
clip_norm = 10000
learning_rate = 1e-3

print("Number of training graphs:", len(train_graph_dicts))
print("Number of test graphs:", len(test_graph_dicts))
print("Batch size:", batch_size)

# --- Model, Loss, and Optimizer ---
model = gnn_model.EncodeProcessDecode(
    node_input_dim=2,
    edge_input_dim=3,
    node_output_size=1,
).to(DEVICE)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_processing_steps_tensor_for_model = torch.tensor(num_processing_steps, dtype=torch.int32).to(DEVICE)

# --- Training and Test Loop ---
best_rho_test = -np.inf
l1_train_hist, l2_train_hist, rho_train_hist = [], [], []
l1_test_hist, l2_test_hist, rho_test_hist = [], [], []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()

    idx = np.arange(len(train_graph_dicts))
    shuffled_train_graphs = [train_graph_dicts[i] for i in idx]
    shuffled_train_targets = [train_targets[i] for i in idx]

    l1s_train_epoch, l2s_train_epoch, rhos_train_epoch = [], [], []

    for i in range(0, len(shuffled_train_graphs), batch_size):
        batch_graphs_list = shuffled_train_graphs[i:i+batch_size]
        batch_targets_list = shuffled_train_targets[i:i+batch_size]

        if not batch_graphs_list:
            continue

        # CORRECT: Normalize the training data just before using it
        normalized_train_graphs = []
        for g in batch_graphs_list:
            g_normalized = g.copy()
            g_normalized["nodes"] = ((g_normalized["nodes"].cpu() - node_mean) / node_std).to(DEVICE)
            if g_normalized["edges"].numel() > 0 and edge_mean is not None and edge_std is not None:
                g_normalized["edges"] = ((g_normalized["edges"].cpu() - edge_mean) / edge_std).to(DEVICE)
            normalized_train_graphs.append(g_normalized)

        batch_graphs_tuple = graph.data_dicts_to_graphs_tuple_pytorch(normalized_train_graphs)
        batch_targets_concat = torch.cat(batch_targets_list, dim=0).to(DEVICE)

        assert batch_graphs_tuple["nodes"].shape[0] == batch_targets_concat.shape[0], "Node count mismatch!"

        # Debugging for Loss
        # import pdb; pdb.set_trace()

        optimizer.zero_grad()
        output_graphs_list = model(batch_graphs_tuple, num_processing_steps_tensor_for_model)
        pred_nodes = output_graphs_list[-1]["nodes"]

        l1_loss_val = F.l1_loss(pred_nodes, batch_targets_concat)
        l2_loss_val = F.mse_loss(pred_nodes, batch_targets_concat)
        rho_val = graph.pearson_corr_pytorch(pred_nodes, batch_targets_concat)

        l2_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        l1s_train_epoch.append(l1_loss_val.item())
        l2s_train_epoch.append(l2_loss_val.item())
        rhos_train_epoch.append(rho_val.item())

    L1_train = np.mean(l1s_train_epoch) if l1s_train_epoch else 0.0
    L2_train = np.mean(l2s_train_epoch) if l2s_train_epoch else 0.0
    rho_train = np.mean(rhos_train_epoch) if rhos_train_epoch else 0.0

    l1_train_hist.append(L1_train)
    l2_train_hist.append(L2_train)
    rho_train_hist.append(rho_train)

    # --- Test Loop ---
    model.eval()
    l1s_test_epoch, l2s_test_epoch, rhos_test_epoch = [], [], []
    test_preds_all, test_targets_all = [], []

    with torch.no_grad():
        for i in range(0, len(test_graph_dicts), batch_size):
            batch_graphs_list = test_graph_dicts[i:i+batch_size]
            batch_targets_list = test_targets[i:i+batch_size]

            if not batch_graphs_list:
                continue

            # CORRECT: Normalize the test data just before using it
            normalized_test_graphs = []
            for g in batch_graphs_list:
                g_normalized = g.copy()
                g_normalized["nodes"] = ((g_normalized["nodes"].cpu() - node_mean) / node_std).to(DEVICE)
                if g_normalized["edges"].numel() > 0 and edge_mean is not None and edge_std is not None:
                    g_normalized["edges"] = ((g_normalized["edges"].cpu() - edge_mean) / edge_std).to(DEVICE)
                normalized_test_graphs.append(g_normalized)

            batch_graphs_tuple = graph.data_dicts_to_graphs_tuple_pytorch(normalized_test_graphs)
            batch_targets_concat = torch.cat(batch_targets_list, dim=0).to(DEVICE)
            output_graphs_list = model(batch_graphs_tuple, num_processing_steps_tensor_for_model)
            pred_nodes_test = output_graphs_list[-1]["nodes"]

            l1s_test_epoch.append(F.l1_loss(pred_nodes_test, batch_targets_concat).item())
            l2s_test_epoch.append(F.mse_loss(pred_nodes_test, batch_targets_concat).item())
            rhos_test_epoch.append(graph.pearson_corr_pytorch(pred_nodes_test, batch_targets_concat).item())
            
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    L1_test = np.mean(l1s_test_epoch) if l1s_test_epoch else 0.0
    L2_test = np.mean(l2s_test_epoch) if l2s_test_epoch else 0.0
    rho_test = np.mean(rhos_test_epoch) if rhos_test_epoch else 0.0

    l1_test_hist.append(L1_test)
    l2_test_hist.append(L2_test)
    rho_test_hist.append(rho_test)
            
    print(f"Epoch {epoch+1:03d}: "
          f" L2_train={L2_train:.4f}, rho_train={rho_train:.4f} | "
          f" L2_test={L2_test:.4f}, rho_test={rho_test:.4f}")

    # --- TensorBoard Logging per Epoch ---
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Hyperparameters/Learning_Rate', current_lr, epoch)
    writer.add_scalars('Loss/L1', {'train': L1_train, 'test': L1_test}, epoch)
    writer.add_scalars('Loss/L2', {'train': L2_train, 'test': L2_test}, epoch)
    writer.add_scalars('Correlation/rho', {'train': rho_train, 'test': rho_test}, epoch)

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            if param.data is not None:
                writer.add_histogram(f'Weights/{name}', param.data, epoch)