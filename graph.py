# graph.py
import torch
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------
# Global configuration variables — set once from main.py
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Data loading utilities
# ----------------------------------------------------------------------

def load_targets(assembly_id, t):
    """
    Reads the NPMNCF file for a specific assembly and timestep.
    """
    if _DATA_PATH is None or _DEVICE is None:
        raise RuntimeError("Global data path or device not set. Call set_global_paths_and_device first.")
    
    fname = os.path.join(_DATA_PATH, f"{assembly_id}_{t}_ball_NPMNCF.tab")
    df = pd.read_csv(fname, sep=r"\s+", header=None, skiprows=2, names=["ball_id", "npmncf"])
    df.sort_values("ball_id", inplace=True)
    df["ball_id"] = pd.to_numeric(df["ball_id"], errors="coerce").astype(float).astype(int)
    df.dropna(subset=["ball_id"], inplace=True)
    return df


def load_scalar_feat(assembly_id, name, t):
    """
    Loads scalar features (like ball_rad, ball_cn, disp_x, etc.)
    from the corresponding .tab files.
    """
    if _DATA_PATH is None:
        raise RuntimeError("Global data path not set. Call set_global_paths_and_device first.")
    
    return pd.read_csv(
        f"{_DATA_PATH}/{assembly_id}_{t}_{name}.tab",
        header=None,
        names=["ball_id", name],
        sep=r"\s+",
        skiprows=2
    )


def load_all_node_data(assembly_id, t):
    """
    Loads all node features for a given timestep.

    Includes:
      - ball_rad : particle radius
      - ball_cn  : coordination number (number of contacts)
      - ball_disp_* and ball_pos_* : for displacement & position
    """
    df = load_scalar_feat(assembly_id, "ball_rad", t)
    df = df.merge(load_scalar_feat(assembly_id, "ball_cn", t), on="ball_id")
    for name in ["ball_disp_x", "ball_disp_y", "ball_disp_z",
                 "ball_pos_x", "ball_pos_y", "ball_pos_z"]:
        df = df.merge(load_scalar_feat(assembly_id, name, t), on="ball_id")
    df.sort_values("ball_id", inplace=True)
    return df


def load_edges(assembly_id, t, id_to_idx):
    """
    Loads edge connections between particles at timestep t,
    mapping ball IDs to node indices.
    """
    if _DATA_PATH is None:
        raise RuntimeError("Global data path not set. Call set_global_paths_and_device first.")
    
    e1 = pd.read_csv(f"{_DATA_PATH}/{assembly_id}_{t}_contact_end1.tab", sep=r"\s+", header=None, skiprows=2)
    e2 = pd.read_csv(f"{_DATA_PATH}/{assembly_id}_{t}_contact_end2.tab", sep=r"\s+", header=None, skiprows=2)

    e1_ids = pd.to_numeric(e1.iloc[:, 1], errors="coerce").dropna().astype(int)
    e2_ids = pd.to_numeric(e2.iloc[:, 1], errors="coerce").dropna().astype(int)

    valid_edges = []
    for a, b in zip(e1_ids, e2_ids):
        if a in id_to_idx and b in id_to_idx:
            valid_edges.append((id_to_idx[a], id_to_idx[b]))
    return set(valid_edges)


# ----------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------

def compute_displacement_features(pos, disp, i, j, max_r):
    """
    Computes normalized edge displacement features between nodes i and j.

    As defined in the paper (Section 2.2):
    - The first element of the edge attribute is contact status.
    - The second and third elements are relative displacements
      along and perpendicular to the contact direction,
      normalized by the maximum particle radius.
    """
    rel_pos = pos[i] - pos[j]
    rel_disp = disp[i] - disp[j]
    norm_rel_pos = np.linalg.norm(rel_pos)
    contact_dir = rel_pos / norm_rel_pos

    disp_along = np.dot(rel_disp, contact_dir)
    disp_perp = np.linalg.norm(rel_disp - disp_along * contact_dir)

    return [disp_along / max_r, disp_perp / max_r]


# ----------------------------------------------------------------------
# Graph creation
# ----------------------------------------------------------------------

def create_all_graphs(assembly_ids):
    """
    Creates a list of graph dictionaries (one per timestep per assembly),
    each containing PyTorch tensors for nodes, edges, senders, and receivers.
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

        # --- First timestep (t = 0) ---
        node_data_0 = load_all_node_data(assembly_id, 0)
        node_data_0["ball_id"] = node_data_0["ball_id"].astype(float).astype(int)
        id_to_idx = {bid: i for i, bid in enumerate(node_data_0["ball_id"])}

        node_data_0["ball_rad"] = node_data_0["ball_rad"].astype(float)
        node_data_0["ball_cn"] = node_data_0["ball_cn"].astype(float)
        max_r = node_data_0["ball_rad"].max()

        # Node attributes: radius (normalized) + coordination number
        nodes_0 = torch.tensor(
            np.stack([
                node_data_0["ball_rad"] / max_r,
                node_data_0["ball_cn"]
            ], axis=1), dtype=torch.float32
        )

        edges_0 = load_edges(assembly_id, 0, id_to_idx)
        pos_0 = node_data_0[["ball_pos_x", "ball_pos_y", "ball_pos_z"]].astype(float).values
        disp_0 = node_data_0[["ball_disp_x", "ball_disp_y", "ball_disp_z"]].astype(float).values

        edge_features = []
        senders, receivers = [], []
        for (i, j) in edges_0:
            senders.append(i)
            receivers.append(j)
            status = 0.0  # existing contact
            disp_feat = compute_displacement_features(pos_0, disp_0, i, j, max_r)
            edge_features.append([status] + disp_feat)

        graph_0 = {
            "nodes": nodes_0.to(_DEVICE),
            "edges": torch.tensor(edge_features, dtype=torch.float32).to(_DEVICE)
            if edge_features else torch.empty(0, 3, dtype=torch.float32).to(_DEVICE),
            "senders": torch.tensor(senders, dtype=torch.int64).to(_DEVICE),
            "receivers": torch.tensor(receivers, dtype=torch.int64).to(_DEVICE),
        }
        graphs_for_assembly.append(graph_0)

        # --- Load raw (unnormalized) NPMNCF targets ---
        targets_df = load_targets(assembly_id, 0)
        npmncf = targets_df["npmncf"].astype(float).values.reshape(-1, 1)
        targets_for_assembly.append(torch.tensor(npmncf, dtype=torch.float32).to(_DEVICE))

        # --- Remaining timesteps (t = 1 ... _TIMESTEPS-1) ---
        for t in range(1, _TIMESTEPS):
            node_data_t = load_all_node_data(assembly_id, t)
            node_data_t["ball_id"] = node_data_t["ball_id"].astype(float).astype(int)
            id_to_idx_t = {bid: i for i, bid in enumerate(node_data_t["ball_id"])}

            node_data_t["ball_rad"] = node_data_t["ball_rad"].astype(float)
            node_data_t["ball_cn"] = node_data_t["ball_cn"].astype(float)

            nodes = torch.tensor(
                np.stack([
                    node_data_t["ball_rad"] / max_r,
                    node_data_t["ball_cn"]
                ], axis=1), dtype=torch.float32
            )

            edges_prev = load_edges(assembly_id, t - 1, id_to_idx_t)
            edges_curr = load_edges(assembly_id, t, id_to_idx_t)
            all_edges = list(edges_prev.union(edges_curr))

            pos_t = node_data_t[["ball_pos_x", "ball_pos_y", "ball_pos_z"]].astype(float).values
            disp_t = node_data_t[["ball_disp_x", "ball_disp_y", "ball_disp_z"]].astype(float).values

            edge_features, senders, receivers = [], [], []
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

            graph = {
                "nodes": nodes.to(_DEVICE),
                "edges": torch.tensor(edge_features, dtype=torch.float32).to(_DEVICE)
                if edge_features else torch.empty(0, 3, dtype=torch.float32).to(_DEVICE),
                "senders": torch.tensor(senders, dtype=torch.int64).to(_DEVICE),
                "receivers": torch.tensor(receivers, dtype=torch.int64).to(_DEVICE),
            }
            graphs_for_assembly.append(graph)

            # --- Raw (unnormalized) targets ---
            targets_df = load_targets(assembly_id, t)
            npmncf = targets_df["npmncf"].astype(float).values.reshape(-1, 1)
            targets_for_assembly.append(torch.tensor(npmncf, dtype=torch.float32).to(_DEVICE))

        all_graphs.extend(graphs_for_assembly)
        all_targets.extend(targets_for_assembly)

    return all_graphs, all_targets


# ----------------------------------------------------------------------
# Graph batching & evaluation
# ----------------------------------------------------------------------

def data_dicts_to_graphs_tuple_pytorch(graph_dicts_list):
    """
    Combines a list of PyTorch graph dictionaries into a single batched graph dictionary.
    """
    if _DEVICE is None:
        raise RuntimeError("Global device not set. Call set_global_paths_and_device first.")

    if not graph_dicts_list:
        dummy_node_dim, dummy_edge_dim = 2, 3
        return {
            "nodes": torch.empty(0, dummy_node_dim, dtype=torch.float32, device=_DEVICE),
            "edges": torch.empty(0, dummy_edge_dim, dtype=torch.float32, device=_DEVICE),
            "senders": torch.empty(0, dtype=torch.int64, device=_DEVICE),
            "receivers": torch.empty(0, dtype=torch.int64, device=_DEVICE),
        }

    batch_nodes, batch_edges, batch_senders, batch_receivers = [], [], [], []
    node_offset = 0

    for graph_dict in graph_dicts_list:
        num_nodes = graph_dict["nodes"].shape[0]
        batch_nodes.append(graph_dict["nodes"])
        batch_edges.append(graph_dict["edges"])
        batch_senders.append(graph_dict["senders"] + node_offset)
        batch_receivers.append(graph_dict["receivers"] + node_offset)
        node_offset += num_nodes

    return {
        "nodes": torch.cat(batch_nodes, dim=0),
        "edges": torch.cat(batch_edges, dim=0),
        "senders": torch.cat(batch_senders, dim=0),
        "receivers": torch.cat(batch_receivers, dim=0),
    }


def pearson_corr_pytorch(x, y):
    """
    Calculates the Pearson correlation coefficient between two tensors.
    """
    x_mean, y_mean = torch.mean(x), torch.mean(y)
    xm, ym = x - x_mean, y - y_mean
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
    return r_num / (r_den + 1e-8)
