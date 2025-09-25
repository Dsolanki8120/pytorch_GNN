# graph.py
import torch
import numpy as np
import pandas as pd
import os
from collections import deque

# Global variables for path, device, and timesteps, to be set from main.py
_DATA_PATH = None
_DEVICE = None
_TIMESTEPS = None

def set_global_paths_and_device(data_path, device, timesteps):
    """
    Sets global configuration variables for functions within this module.
    """
    global _DATA_PATH, _DEVICE, _TIMESTEPS
    _DATA_PATH = data_path
    _DEVICE = device
    _TIMESTEPS = timesteps
    print(f"Graph utilities: Data path set to {_DATA_PATH}, device set to {_DEVICE}")


def load_targets(assembly_id,t):
    """
    Reads the NPMNCF file for a specific assembly and timestep.
    """
    if _DATA_PATH is None or _DEVICE is None:
        raise RuntimeError("Global data path or device not set. Call set_global_paths_and_device first.")
    
    fname = os.path.join(_DATA_PATH, f"{assembly_id}_{t}_ball_NPMNCF.tab")
    df = pd.read_csv(fname, sep='\s+', header=None, skiprows=2, names=["ball_id", "npmncf"])
    df.sort_values("ball_id", inplace=True)
    df["ball_id"] = pd.to_numeric(df["ball_id"], errors='coerce').astype(float).astype(int)
    df.dropna(subset=["ball_id"], inplace=True)
    return df
def load_scalar_feat(assembly_id,name,t):
    """
    For files like 0_ball_rad.tab, which are space-separated and have 2 header lines.
    """
    if _DATA_PATH is None:
        raise RuntimeError("Global data path not set. Call set_global_paths_and_device first.")
    
    return pd.read_csv(
        f"{_DATA_PATH}/{assembly_id}_{t}_{name}.tab",
        header=None,
        names=["ball_id", name],
        sep='\s+',
        skiprows=2
    )


def load_all_node_data(assembly_id,t):
    """
    Loads all relevant scalar features for nodes at a given timestep.
    """
    df = load_scalar_feat(assembly_id,"ball_rad", t)
    # df = df.merge(load_scalar_feat(assembly_id,"ball_cn", t), on="ball_id")
    for name in ["ball_disp_x", "ball_disp_y", "ball_disp_z",
                 "ball_pos_x", "ball_pos_y", "ball_pos_z"]:
        df = df.merge(load_scalar_feat(assembly_id,name, t), on="ball_id")
    df.sort_values("ball_id", inplace=True)
    return df


def load_edges(assembly_id, t, id_to_idx):
    """
    Loads edges for a given timestep, mapping ball_ids to current node indices.
    """
    if _DATA_PATH is None:
        raise RuntimeError("Global data path not set. Call set_global_paths_and_device first.")
    
    e1 = pd.read_csv(f"{_DATA_PATH}/{assembly_id}_{t}_contact_end1.tab", sep='\s+', header=None, skiprows=2)
    e2 = pd.read_csv(f"{_DATA_PATH}/{assembly_id}_{t}_contact_end2.tab", sep='\s+', header=None, skiprows=2)

    # FIX: Use pd.to_numeric with errors='coerce' to handle invalid values
    # Then, drop the NaN values before converting to int
    e1_ids = pd.to_numeric(e1.iloc[:, 1], errors='coerce').dropna().astype(int)
    e2_ids = pd.to_numeric(e2.iloc[:, 1], errors='coerce').dropna().astype(int)

    valid_edges = []
    for a, b in zip(e1_ids, e2_ids):
        if a in id_to_idx and b in id_to_idx:
            valid_edges.append((id_to_idx[a], id_to_idx[b]))
    return set(valid_edges)


def compute_displacement_features(pos, disp, i, j, max_r):
    """
    Computes displacement features for an edge.
    Input pos and disp are expected to be NumPy arrays.
    """
    rel_pos = pos[i] - pos[j]
    rel_disp = disp[i] - disp[j]

    norm_rel_pos = np.linalg.norm(rel_pos)

    contact_dir = rel_pos / norm_rel_pos

    disp_along = np.dot(rel_disp, contact_dir)
    disp_perp = np.linalg.norm(rel_disp - disp_along * contact_dir)

    return [disp_along / max_r, disp_perp / max_r]

def create_all_graphs(assembly_ids):
    """
    Creates a list of graph dictionaries for all assemblies and timesteps.
    Each graph dictionary contains PyTorch tensors.
    """
    if _DATA_PATH is None or _DEVICE is None or _TIMESTEPS is None:
        raise RuntimeError("Global data path, device, or timesteps not set. Call set_global_paths_and_device first.")

    print("create_all_graphs called")

    all_graphs = []
    all_targets = []

    for assembly_id in assembly_ids:
        print(f"Building graphs for assembly: {assembly_id}")

        graphs_for_assembly = []
        targets_for_assembly = []
        senders = []
        receivers = []

        # Load node data for the first timestep (t=0)
        node_data_0 = load_all_node_data(assembly_id, 0)
        
        # FIX: Convert the 'ball_id' column to float then int to handle scientific notation
        node_data_0["ball_id"] = node_data_0["ball_id"].astype(float).astype(int)
        id_to_idx = {bid: i for i, bid in enumerate(node_data_0["ball_id"])}

        # FIX: Convert relevant columns to float before any calculations
        node_data_0["ball_rad"] = node_data_0["ball_rad"].astype(float)
        # node_data_0["ball_cn"] = node_data_0["ball_cn"].astype(float)
        
        max_r = node_data_0["ball_rad"].max()

        nodes_0 = torch.tensor(
            np.stack([
                node_data_0["ball_rad"] / max_r,
                # node_data_0["ball_cn"]
            ], axis=1), dtype=torch.float32)

        edges_0 = load_edges(assembly_id, 0, id_to_idx)
        
        pos_0 = node_data_0[["ball_pos_x", "ball_pos_y", "ball_pos_z"]].astype(float).values
        disp_0 = node_data_0[["ball_disp_x", "ball_disp_y", "ball_disp_z"]].astype(float).values

        edge_features = []
        for (i, j) in edges_0:
            senders.append(i)
            receivers.append(j)
            status = 0.0
            disp_feat = compute_displacement_features(pos_0, disp_0, i, j, max_r)
            edge_features.append([status] + disp_feat)

        graph_0 = {
            "nodes": nodes_0.to(_DEVICE),
            "edges": torch.tensor(edge_features, dtype=torch.float32).to(_DEVICE) if edge_features else torch.empty(0, 3, dtype=torch.float32).to(_DEVICE),
            "senders": torch.tensor(senders, dtype=torch.int64).to(_DEVICE),
            "receivers": torch.tensor(receivers, dtype=torch.int64).to(_DEVICE)
        }
        graphs_for_assembly.append(graph_0)
        
        targets_df = load_targets(assembly_id, 0)
        targets_df["ball_id"] = targets_df["ball_id"].astype(float).astype(int)
        
        targets_for_assembly.append(torch.tensor(targets_df["npmncf"].values.reshape(-1, 1), dtype=torch.float32).to(_DEVICE))

        # Build graphs for subsequent timesteps (t=1 to _TIMESTEPS-1)
        for t in range(1, _TIMESTEPS):
            senders.clear()
            receivers.clear()

            node_data_t = load_all_node_data(assembly_id, t)

            # FIX: Convert 'ball_id' column to float then int to handle scientific notation
            node_data_t["ball_id"] = node_data_t["ball_id"].astype(float).astype(int)
            id_to_idx_t = {bid: i for i, bid in enumerate(node_data_t["ball_id"])}

            # FIX: Convert relevant columns to float before any calculations
            node_data_t["ball_rad"] = node_data_t["ball_rad"].astype(float)
            # node_data_t["ball_cn"] = node_data_t["ball_cn"].astype(float)
            
            nodes = torch.tensor(
                np.stack([
                    node_data_t["ball_rad"] / max_r,
                    # node_data_t["ball_cn"]
                ], axis=1), dtype=torch.float32)

            edges_prev = load_edges(assembly_id, t - 1, id_to_idx_t)
            edges_curr = load_edges(assembly_id, t, id_to_idx_t)
            all_edges = list(edges_prev.union(edges_curr))

            pos_t = node_data_t[["ball_pos_x", "ball_pos_y", "ball_pos_z"]].astype(float).values
            disp_t = node_data_t[["ball_disp_x", "ball_disp_y", "ball_disp_z"]].astype(float).values


            edge_features = []
            for (i, j) in all_edges:
                if (i, j) in edges_prev and (i, j) in edges_curr:
                    status = 0.0
                elif (i, j) in edges_prev:
                    status = -1.0
                else:
                    status = 1.0


                if i < pos_t.shape[0] and j < pos_t.shape[0]:
                    disp_feat = compute_displacement_features(pos_t, disp_t, i, j, max_r)
                    edge_features.append([status] + disp_feat)
                    senders.append(i)
                    receivers.append(j)
                else:
                    print(f"Skipping edge ({i},{j}) - index out of bounds for pos_t of shape {pos_t.shape}")

            graph = {
                "nodes": nodes.to(_DEVICE),
                "edges": torch.tensor(edge_features, dtype=torch.float32).to(_DEVICE) if edge_features else torch.empty(0, 3, dtype=torch.float32).to(_DEVICE),
                "senders": torch.tensor(senders, dtype=torch.int64).to(_DEVICE),
                "receivers": torch.tensor(receivers, dtype=torch.int64).to(_DEVICE)
            }
            graphs_for_assembly.append(graph)
            
            targets_df = load_targets(assembly_id, t)
            targets_df["ball_id"] = pd.to_numeric(targets_df["ball_id"], errors='coerce').dropna().astype(int)
            targets_for_assembly.append(torch.tensor(targets_df["npmncf"].values.reshape(-1, 1), dtype=torch.float32).to(_DEVICE))

        all_graphs.extend(graphs_for_assembly)
        all_targets.extend(targets_for_assembly)

    return all_graphs, all_targets


def data_dicts_to_graphs_tuple_pytorch(graph_dicts_list):
    """
    Combines a list of PyTorch graph dictionaries into a single batched graph dictionary.
    Assumes all tensors in input graph_dicts are already on the correct DEVICE.
    """
    if _DEVICE is None:
        raise RuntimeError("Global device not set. Call set_global_paths_and_device first.")

    if not graph_dicts_list:
        dummy_node_dim = 2
        dummy_edge_dim = 3
  
        # Returns 2D empty tensors for consistency in shape even for empty batches
        return {
            "nodes": torch.empty(0, dummy_node_dim, dtype=torch.float32, device=_DEVICE),
            "edges": torch.empty(0, dummy_edge_dim, dtype=torch.float32, device=_DEVICE),
            "senders": torch.empty(0, dtype=torch.int64, device=_DEVICE),
            "receivers": torch.empty(0, dtype=torch.int64, device=_DEVICE),
        }

    batch_nodes = []
    batch_edges = []
    batch_senders = []
    batch_receivers = []
    node_offset = 0

    for graph_dict in graph_dicts_list:
        num_nodes_in_graph = graph_dict["nodes"].shape[0]

        batch_nodes.append(graph_dict["nodes"])
        batch_edges.append(graph_dict["edges"])
        batch_senders.append(graph_dict["senders"] + node_offset)
        batch_receivers.append(graph_dict["receivers"] + node_offset)

        node_offset += num_nodes_in_graph
    batched_graph = {
        "nodes": torch.cat(batch_nodes, dim=0),
        "edges": torch.cat(batch_edges, dim=0),
        "senders": torch.cat(batch_senders, dim=0),
        "receivers": torch.cat(batch_receivers, dim=0),
        
    }

    return batched_graph


def pearson_corr_pytorch(x, y):
    """
    Calculates Pearson correlation coefficient between two tensors.
    """
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    xm, ym = x - x_mean, y - y_mean
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
    return r_num / (r_den + 1e-8)


