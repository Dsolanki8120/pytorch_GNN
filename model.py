import torch
import torch.nn as nn
import torch.nn.functional as F

_DEVICE = None

def set_device_for_model(device):
    """Sets the global device for operations within this module."""
    global _DEVICE
    _DEVICE = device


# -------------------------------------------------
# 1. MLP builder (2 layers × 64 neurons, paper spec)
# -------------------------------------------------
def make_mlp_model(input_dim, output_dim, hidden_size=64, num_hidden_layers=2):
    layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
    layers += [nn.Linear(hidden_size, output_dim)]
    return nn.Sequential(*layers)


# -------------------------------------------------
# 2. Edge and Node update blocks (paper Appendix)
# -------------------------------------------------
class EdgeUpdate(nn.Module):
    """
    Implements Eq. (7)-(8):
        B_e = [F_v R_r ; F_v R_s ; F_e]
        F'_e = φ_e(B_e)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim
        self.mlp = make_mlp_model(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_features, senders, receivers):
        sender_feats = node_features[senders]
        receiver_feats = node_features[receivers]
        edge_input = torch.cat([receiver_feats, sender_feats, edge_features], dim=-1)
        out = self.mlp(edge_input)
        return self.norm(out)


class NodeUpdate(nn.Module):
    """
    Implements Eq. (9)-(10):
        B_n = [F_v ; F'_e R_r^T]
        F'_v = φ_n(B_n)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        input_dim = node_dim + edge_dim
        self.mlp = make_mlp_model(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_features, receivers, num_nodes):
        agg_edges = torch.zeros(num_nodes, edge_features.shape[1],
                                device=_DEVICE, dtype=edge_features.dtype)
        agg_edges.index_add_(0, receivers, edge_features)

        # Mean aggregation to normalize by node degree
        deg = torch.bincount(receivers, minlength=num_nodes).clamp(min=1).unsqueeze(-1)
        agg_edges = agg_edges / deg

        node_input = torch.cat([node_features, agg_edges], dim=-1)
        out = self.mlp(node_input)
        return self.norm(out)


# -------------------------------------------------
# 3. Encode–Process–Decode Network (paper’s Fig. 2)
# -------------------------------------------------
class EncodeProcessDecode(nn.Module):
    """
    Implements:
        G_0 = Encoder(G_in)
        G_n = Core(G_{n-1}; G_0) with skip connections
        G_out = Decoder(G_N)
    """
    def __init__(self, node_input_dim, edge_input_dim, node_output_size,
                 hidden_dim=64, num_processing_steps=7):
        super().__init__()
        self.num_processing_steps = num_processing_steps

        # Encoder
        self.edge_encoder = make_mlp_model(2 * node_input_dim + edge_input_dim, hidden_dim)
        self.node_encoder = make_mlp_model(node_input_dim + hidden_dim, hidden_dim)

        # Core (shared across steps)
        self.edge_core = EdgeUpdate(hidden_dim, hidden_dim, hidden_dim)
        self.node_core = NodeUpdate(hidden_dim, hidden_dim, hidden_dim)

        # Decoder
        self.node_decoder = make_mlp_model(hidden_dim, node_output_size)

        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_dict, num_processing_steps_tensor):
        num_processing_steps = num_processing_steps_tensor.item()
        nodes = graph_dict["nodes"]
        edges = graph_dict["edges"]
        senders = graph_dict["senders"]
        receivers = graph_dict["receivers"]

        num_nodes = nodes.shape[0]

        # -----------------
        # Encode phase
        # -----------------
        sender_nodes = nodes[senders]
        receiver_nodes = nodes[receivers]
        edge_input = torch.cat([receiver_nodes, sender_nodes, edges], dim=-1)
        encoded_edges = self.edge_norm(self.edge_encoder(edge_input))

        agg_edges = torch.zeros(num_nodes, encoded_edges.shape[1],
                                device=_DEVICE, dtype=encoded_edges.dtype)
        agg_edges.index_add_(0, receivers, encoded_edges)
        deg = torch.bincount(receivers, minlength=num_nodes).clamp(min=1).unsqueeze(-1)
        agg_edges = agg_edges / deg

        node_input = torch.cat([nodes, agg_edges], dim=-1)
        encoded_nodes = self.node_norm(self.node_encoder(node_input))

        latent_nodes = encoded_nodes.clone()
        latent_edges = encoded_edges.clone()

        # -----------------
        # Process (Message passing)
        # -----------------
        for _ in range(num_processing_steps):
            updated_edges = self.edge_core(latent_nodes, latent_edges, senders, receivers)
            latent_edges = latent_edges + updated_edges  # skip connection (Eq. 2)

            updated_nodes = self.node_core(latent_nodes, latent_edges, receivers, num_nodes)
            latent_nodes = latent_nodes + updated_nodes  # skip connection (Eq. 2)

        # -----------------
        # Decode
        # -----------------
        decoded_nodes = self.node_decoder(latent_nodes)

        return [{"nodes": decoded_nodes}]
