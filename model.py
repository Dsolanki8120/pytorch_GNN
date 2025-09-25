# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEVICE = None

def set_device_for_model(device):
    """Sets the global device for operations within this module."""
    global _DEVICE
    _DEVICE = device

# --- Global Constants for Model Architecture ---
LATENT_SIZE = 64 # Internal feature dimension for hidden layers
NUM_LAYERS = 2  # Number of linear layers within each MLP sub-block (e.g., encoder, processor)

# --- Helper for creating MLP models ---
def make_mlp_model(input_dim, output_dim):
    """
    Instantiates a simple Multi-Layer Perceptron (MLP) with ReLU activations.
    Maps input_dim -> LATENT_SIZE (hidden) -> ... -> output_dim.
    """
    mlp_layers = []
    # Input layer
    mlp_layers.append(nn.Linear(input_dim, LATENT_SIZE))
    mlp_layers.append(nn.LeakyReLU(negative_slope = 0.01))
    # Hidden layers
    for _ in range(NUM_LAYERS):
        mlp_layers.append(nn.Linear(LATENT_SIZE, LATENT_SIZE))
        mlp_layers.append(nn.LeakyReLU(negative_slope = 0.01))
    
    # Output layer (only if final output_dim differs from LATENT_SIZE)
    mlp_layers.append(nn.Linear(LATENT_SIZE, output_dim))

    return nn.Sequential(*mlp_layers)



class EdgeEncoderBlock(nn.Module):
    def __init__(self,initial_node_dim,intial_edge_dim,LATENT_SIZE):
        super().__init__()
        input_dim= 2*initial_node_dim + intial_edge_dim
        self.mlp= make_mlp_model(input_dim,LATENT_SIZE)
        self.norm = nn.LayerNorm(LATENT_SIZE)  # added LayerNorm


    def forward(self,edge_features,sender_node_features,reciever_node_features):
        edge_input= torch.cat([edge_features,sender_node_features,reciever_node_features],dim=-1)
        return self.norm(self.mlp(edge_input))
    

class NodeEncoderBlock(nn.Module):
    def __init__(self,initial_node_dim,agg_edg_feature_to_node,LATENT_SIZE):
        super().__init__()
        input_dim= initial_node_dim + agg_edg_feature_to_node
        self.mlp= make_mlp_model(input_dim,LATENT_SIZE)
        self.norm = nn.LayerNorm(LATENT_SIZE)   # added LayerNorm

    
    def forward(self,initial_node_dim,agg_edg_feature_to_node):
        node_input= torch.cat([initial_node_dim,agg_edg_feature_to_node],dim=-1)
        return self.norm(self.mlp(node_input))




class EdgeBlock(nn.Module):
    """
    Updates edge features based on sender, receiver node features, edge features, and global features.
    """
    def __init__(self, node_dim, edge_dim,  edge_output_size):
        super().__init__()
        # Input features for edge update MLP: sender_node_feat + receiver_node_feat + edge_feat + global_feat
        input_dim = 2 * node_dim + edge_dim + node_dim*2 + edge_dim  
        self.mlp = make_mlp_model(input_dim, edge_output_size)
        self.norm = nn.LayerNorm(edge_output_size)


    def forward(self, edge_features, sender_node_features, receiver_node_features, g0_edge_features, g0_sender_features, g0_receiver_features):
        
        edge_input = torch.cat([
            sender_node_features, receiver_node_features, edge_features,
            g0_sender_features, g0_receiver_features, g0_edge_features
                                                                       ], dim=-1)
        
        mlp_output = self.mlp(edge_input)

        return self.norm(mlp_output)
        
     

class NodeBlock(nn.Module):
    """
    Updates node features based on node features, aggregated edge features, and global features.
    """
    def __init__(self, node_dim, edge_dim, node_output_size):
        super().__init__()
        # Input features for node update MLP: node_feat + aggregated_edge_feat + global_feat
        # edge_dim here refers to the dimension of the aggregated edge features
        input_dim = input_dim = node_dim + edge_dim + node_dim + edge_dim

        self.mlp = make_mlp_model(input_dim, node_output_size)
        self.norm = nn.LayerNorm(node_output_size)
      
        

    def forward(self, node_features, aggregated_edge_features,g0_node_features, g0_agg_edges):
                
        node_input = torch.cat([
            node_features, aggregated_edge_features,
            g0_node_features, g0_agg_edges
        ], dim=-1)
        
        # Only append global features if the block was initialized to use them (global_dim is not None)
        # AND the actual global_features tensor is provided and has elements.

        mlp_output = self.mlp(node_input)
            
        return self.norm(mlp_output)





class EncodeProcessDecode(nn.Module):
    """
    Full encode-process-decode GNN model.
    The model includes three components: Encoder, Core (processor), and Decoder.
    """
    def __init__(self,node_input_dim,edge_input_dim,node_output_size):
        super().__init__()

        # Encoders: Map raw input features to latent (hidden) dimension (LATENT_SIZE)
        self.edge_encoder = EdgeEncoderBlock(node_input_dim, edge_input_dim, LATENT_SIZE)
        self.node_encoder = NodeEncoderBlock(node_input_dim, LATENT_SIZE, LATENT_SIZE)
        
        # Processor blocks: Perform message passing and updates in latent space
        # global_hidden_size is passed to blocks; it will be None if global_output_size is None
        self.edge_processor = EdgeBlock(LATENT_SIZE, LATENT_SIZE,  LATENT_SIZE)
        self.node_processor = NodeBlock(LATENT_SIZE, LATENT_SIZE,  LATENT_SIZE)
      
        # Decoders: Map latent features to desired output dimensions
        self.node_decoder =  make_mlp_model(LATENT_SIZE, node_output_size)
        # self.edge_decoder = make_mlp_model(edge_hidden_size, edge_output_size)if edge_output_size else None

        

       

    def forward(self, graph_dict, num_processing_steps_tensor): # num_processing_steps is now a tensor
        # Convert the input tensor back to an integer for use in the loop
        num_processing_steps = num_processing_steps_tensor.item() 

        # Unpack graph components from the input dictionary
        nodes = graph_dict["nodes"]
        edges = graph_dict["edges"]
        senders = graph_dict["senders"]
        receivers = graph_dict["receivers"]
        
      

        # 1. Encode: Map raw features to latent space
        sender_nodes_for_edges = nodes[senders]
        receiver_nodes_for_edges = nodes[receivers]
        encode_edges = self.edge_encoder(edges,sender_nodes_for_edges,receiver_nodes_for_edges)

        agg_edges_to_nodes = torch.zeros(
                nodes.shape[0], encode_edges.shape[1],
                dtype=encode_edges.dtype, device=_DEVICE
            )
        agg_edges_to_nodes.index_add_(0, receivers, encode_edges)
        encode_nodes = self.node_encoder(nodes,agg_edges_to_nodes)

        latent_nodes = encode_nodes
        latent_edges = encode_edges

        
     

        # Ensure _DEVICE is set before using it for tensor creation
        if _DEVICE is None:
            raise RuntimeError("DEVICE not set for model. Call model.set_device_for_model(device) first.")


        # 2. Process (Message Passing Loop): Iteratively refine latent features
        for step_idx in range(num_processing_steps):
            # Edge update
            sender_nodes = latent_nodes[senders]
            receiver_nodes = latent_nodes[receivers]
            g0_sender = encode_nodes[senders]
            g0_receiver = encode_nodes[receivers]
            latent_edges = self.edge_processor( latent_edges, sender_nodes, receiver_nodes,encode_edges, g0_sender, g0_receiver) 
            


            # Aggregate edge features to nodes
            agg_edges_to_nodes = torch.zeros(
                latent_nodes.shape[0], latent_edges.shape[1],
                dtype=latent_edges.dtype, device=_DEVICE
            )
            agg_edges_to_nodes.index_add_(0, receivers, latent_edges)

            g0_agg_edges = torch.zeros_like(agg_edges_to_nodes)
            g0_agg_edges.index_add_(0, receivers, encode_edges)

            # Node update
            latent_nodes = self.node_processor(  latent_nodes, agg_edges_to_nodes,encode_nodes, g0_agg_edges)
           

            
            
        # 3. Decode: Map final latent features to desired output space
        decoded_nodes = self.node_decoder(latent_nodes)
        

        # Return a list containing the final decoded graph components
        final_output_graph = {
            "nodes": decoded_nodes,
        }
        return [final_output_graph]


